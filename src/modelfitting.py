"""
This module contains functions for fitting a model spectrum with data using downhill simplex method.
Does not fit wavelength coeffs in modelspec_template(). 
"""

import numpy as np
import scipy.constants as const


def errfunc(*arg, **kw):
    """Generic function to give the chi-squared error on a generic
        function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

      OR:
       
       (fitparams, function, arg1, arg2, ... , depvar, weights, kw)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.

      If the last argument is of type dict, it is assumed to be a set
      of keyword arguments: this will be added to errfunc2's direct
      keyword arguments, and will then be passed to the fitting
      function **kw.  This is necessary for use with various fitting
      and sampling routines (e.g., kapteyn.kmpfit and emcee.sampler)
      which do not allow keyword arguments to be explicitly passed.
      So, we cheat!  Note that any keyword arguments passed in this
      way will overwrite keywords of the same names passed in the
      standard, Pythonic, way.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      gaussprior -- list of 2-tuples (or None values), same length as "fitparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  Here in
                   :func:`devfunc`, we _scale_ the error-weighted
                   deviates such that the resulting chi-squared will
                   increase by the desired amount.

      uniformprior -- list of 2-tuples (or 'None's), same length as "fitparams."
                   The i^th tuple (lo_i, hi_i) imposes a uniform prior
                   on the i^th parameter p_i by requiring that it lie
                   within the specified "high" and "low" limits.  We
                   do this (imprecisely) by multiplying the resulting
                   deviates by 1e9 for each parameter outside its
                   limits.

      ngaussprior -- list of 3-tuples of Numpy arrays.
                   Each tuple (j_ind, mu, cov) imposes a multinormal
                   Gaussian prior on the parameters indexed by
                   'j_ind', with mean values specified by 'mu' and
                   covariance matrix 'cov.' This is the N-dimensional
                   generalization of the 'gaussprior' option described
                   above. Here in :func:`devfunc`, we _scale_ the
                   error-weighted deviates such that the resulting
                   chi-squared will increase by the desired amount.

                   For example, if parameters 0 and 3 are to be
                   jointly constrained (w/unity means), set: 
                     jparams = np.array([0, 3])
                     mu = np.array([1, 1])
                     cov = np.array([[1, .9], [9., 1]])
                     ngaussprior=[[jparams, mu, cov]]  # Double brackets are key!

      scaleErrors -- bool
                   If True, instead of chi^2 we return:
                     chi^2 / s^2  +  2N ln(s)
                   Where 's' is the first input parameter (pre-pended
                   to those used for the specified function) and N the
                   number of datapoints.
   

                   In this case, the first element of 'fitparams'
                   ("s") is used to rescale the measurement
                   uncertainties. Thus weights --> weights/s^2, and
                   chi^2 --> 2 N log(s) + chi^2/s^2 (for N data points).  


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))

    """
    # 2009-12-15 13:39 IJC: Created
    # 2010-11-23 16:25 IJMC: Added 'testfinite' flag keyword
    # 2011-06-06 10:52 IJMC: Added 'useindepvar' flag keyword
    # 2011-06-24 15:03 IJMC: Added multi-function (npars) and
    #                        jointpars support.
    # 2011-06-27 14:34 IJMC: Flag-catching for multifunc calling
    # 2012-03-23 18:32 IJMC: testfinite and useindepvar are now FALSE
    #                        by default.
    # 2012-05-01 01:04 IJMC: Adding surreptious keywords, and GAUSSIAN
    #                        PRIOR capability.
    # 2012-05-08 16:31 IJMC: Added NGAUSSIAN option.
    # 2012-10-16 09:07 IJMC: Added 'uniformprior' option.
    # 2013-02-26 11:19 IJMC: Reworked return & concatenation in 'npars' cases.
    # 2013-03-08 12:54 IJMC: Added check for chisq=0 in penalty-factor cases.
    # 2013-04-30 15:33 IJMC: Added C-based chi-squared calculator;
    #                        made this function separate from devfunc.
    # 2013-07-23 18:32 IJMC: Now 'ravel' arguments for C-based function.
    # 2013-10-12 23:47 IJMC: Added 'jointpars1' keyword option.
    # 2014-05-02 11:45 IJMC: Added 'scaleErrors' keyword option..

    params = np.array(arg[0], copy=False)
    #if 'wrapped_joint_params' in kw:
    #    params = unwrap_joint_params(params, kw['wrapped_joint_params'])

    if isinstance(arg[-1], dict): 
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass


    if len(arg)==2:
        chisq = errfunc(params, *arg[1], **kw)

    else:
        testfinite = ('testfinite' in kw) and kw['testfinite']
        if not ('useindepvar' in kw):
            kw['useindepvar'] = False

        # Keep fixed pairs of joint parameters:
        if ('jointpars1' in kw):
            jointpars1 = kw['jointpars1']
            for jointpar1 in jointpars1:
                params[jointpar1[1]] = params[jointpar1[0]]


        if ('gaussprior' in kw) and kw['gaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_gaussprior =  kw['gaussprior']
            gaussprior = []
            for pair in temp_gaussprior:
                if pair is None:
                    gaussprior.append([0, np.inf])
                else:
                    gaussprior.append(pair)
        else:
            gaussprior = None

        if ('uniformprior' in kw):
            # If any priors are None, redefine them:
            temp_uniformprior =  kw['uniformprior']
            uniformprior = []
            for pair in temp_uniformprior:
                if pair is None:
                    uniformprior.append([-np.inf, np.inf])
                else:
                    uniformprior.append(pair)
        else:
            uniformprior = None

        if ('ngaussprior' in kw) and kw['ngaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_ngaussprior =  kw['ngaussprior']
            ngaussprior = []
            for triplet in temp_ngaussprior:
                if len(triplet)==3:
                    ngaussprior.append(triplet)
        else:
            ngaussprior = None


        if ('npars' in kw):
            npars = kw['npars']
            chisq = 0.0
            # Excise "npars" kw for recursive calling:
            lower_kw = kw.copy()
            junk = lower_kw.pop('npars')

            # Keep fixed pairs of joint parameters:
            if ('jointpars' in kw):
                jointpars = kw['jointpars']
                for jointpar in jointpars:
                    params[jointpar[1]] = params[jointpar[0]]
                #pdb.set_trace()

            for ii in range(len(npars)):
                i0 = sum(npars[0:ii])
                i1 = i0 + npars[ii]
                these_params = arg[0][i0:i1]
                these_params, lower_kw = subfit_kw(arg[0], kw, i0, i1)
                chisq  += errfunc(these_params, *arg[ii+1], **lower_kw)

            return chisq

        else: # Single function-fitting
            depvar = arg[-2]
            weights = arg[-1]

            if not kw['useindepvar']:  # Standard case:
                functions = arg[1]
                helperargs = arg[2:len(arg)-2]
            else:                      # Obsolete, deprecated case:
                functions = arg[1] 
                helperargs = arg[2:len(arg)-3]
                indepvar = arg[-3]

        if testfinite:
            finiteind = np.isfinite(indepvar) * np.isfinite(depvar) * np.isfinite(weights)
            indepvar = indepvar[finiteind]
            depvar = depvar[finiteind]
            weights = weights[finiteind]

        doScaleErrors = 'scaleErrors' in kw and kw['scaleErrors']==True
        if doScaleErrors:
            #pdb.set_trace()
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params[1:],)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params[1:],)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            chisq = (weights*((model-depvar))**2).sum()
            chisq = chisq/params[0]**2 + 2*depvar.size*np.log(np.abs(params[0]))

        else:
            if not kw['useindepvar'] or arg[1].__name__=='multifunc' or \
                    arg[1].__name__=='sumfunc':
                model = functions(*((params,)+helperargs))
            else:  # i.e., if useindepvar is True -- old, deprecated usage:
                model = functions(*((params,)+helperargs + (indepvar,)))

            # Compute the weighted residuals:
            chisq = (weights*(model-depvar)**2).sum()
            

        # Compute 1D and N-D gaussian, and uniform, prior penalties:
        additionalChisq = 0.
        if gaussprior is not None:
            additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for \
                                   param0, gprior in zip(params, gaussprior)])

        if ngaussprior is not None:
            for ind, mu, cov in ngaussprior:
                dvec = params[ind] - mu
                additionalChisq += \
                    np.dot(dvec.transpose(), np.dot(np.linalg.inv(cov), dvec))

        if uniformprior is not None:
            for param0, uprior in zip(params, uniformprior):
                if (param0 < uprior[0]) or (param0 > uprior[1]):
                    chisq *= 1e6

        # Scale up the residuals so as to impose priors in chi-squared
        # space:
        chisq += additionalChisq
    
    return chisq


def fmin(func, x0, args=(), kw=dict(),  xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, zdelt = 0.00025, nonzdelt = 0.05, 
         holdfixed=None):
    """Minimize a function using the downhill simplex algorithm -- now with KEYWORDS.

    :Parameters:

      func : callable func(x,*args)
          The objective function to be minimized.
      x0 : ndarray
          Initial guess.
      args : tuple
          Extra arguments passed to func, i.e. ``f(x,*args)``.
      callback : callable
          Called after each iteration, as callback(xk), where xk is the
          current parameter vector.

    :Returns: (xopt, {fopt, iter, funcalls, warnflag})

      xopt : ndarray
          Parameter that minimizes function.
      fopt : float
          Value of function at minimum: ``fopt = func(xopt)``.
      iter : int
          Number of iterations performed.
      funcalls : int
          Number of function calls made.
      warnflag : int
          1 : Maximum number of function evaluations made.
          2 : Maximum number of iterations reached.
      allvecs : list
          Solution at each iteration.

    *Other Parameters*:

      xtol : float
          Relative error in xopt acceptable for convergence.
      ftol : number
          Relative error in func(xopt) acceptable for convergence.
      maxiter : int
          Maximum number of iterations to perform.
      maxfun : number
          Maximum number of function evaluations to make [200*len(x0)]
      full_output : bool
          Set to True if fval and warnflag outputs are desired.
      disp : bool
          Set to True to print convergence messages.
      retall : bool
          Set to True to return list of solutions at each iteration.
      zdelt : number
          Set the initial stepsize for x0[k] equal to zero
      nonzdelt : number
          Set the initial stepsize for x0[k] nonzero
      holdfixed : sequence
          Indices of x0 to hold fixed (e.g., [1, 2, 4])


    :TBD:  gprior : tuple or sequence of tuples
          Set a gaussian prior on the indicated parameter, such that
          chisq += ((x0[p] - val)/unc_val)**2, where the parameters
          are defined by the tuple gprior=(param, val, unc_val)

    :Notes:

        Uses a Nelder-Mead simplex algorithm to find the minimum of
        function of one or more variables.

    """
    # 2011-04-13 14:26 IJMC: Adding Keyword option
    # 2011-05-11 10:48 IJMC: Added the zdelt and nonzdelt options
    # 2011-05-30 15:36 IJMC: Added the holdfixed option

    def wrap_function(function, args, **kw):
        ncalls = [0]
        def function_wrapper(x):
            ncalls[0] += 1
            return function(x, *args, **kw)
        return ncalls, function_wrapper

    # Set up holdfixed arrays
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        holdsome = True
    else:
        holdsome = False
    

    fcalls, func = wrap_function(func, args, **kw)
    x0 = np.asfarray(x0).flatten()
    xoriginal = x0.copy()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
    one2np1 = range(1,N+1)

    if rank == 0:
        sim = np.zeros((N+1,), dtype=x0.dtype)
    else:
        sim = np.zeros((N+1,N), dtype=x0.dtype)
    fsim = np.zeros((N+1,), float)
    sim[0] = x0
    if retall:
        allvecs = [sim[0]]

    fsim[0] = func(x0)
    for k in range(0,N):
        y = np.array(x0,copy=True)
        if y[k] != 0:
            y[k] = (1+nonzdelt)*y[k]
        else:
            y[k] = zdelt
        if holdsome and k in holdfixed:
            y[k] = xoriginal[k]
        sim[k+1] = y
        f = func(y)
        fsim[k+1] = f

    ind = np.argsort(fsim)
    fsim = np.take(fsim,ind,0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim,ind,0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):

        if (max(np.ravel(abs(sim[1:]-sim[0]))) <= xtol \
            and max(abs(fsim[0]-fsim[1:])) <= ftol):
            break

        xbar = np.add.reduce(sim[:-1],0) / N
        xr = (1+rho)*xbar - rho*sim[-1]
        if holdsome:
            xr[holdfixed] = xoriginal[holdfixed]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
            if holdsome:
                xe[holdfixed] = xoriginal[holdfixed]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else: # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else: # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1+psi*rho)*xbar - psi*rho*sim[-1]
                    if holdsome:
                        xc[holdfixed] = xoriginal[holdfixed]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink=1
                else:
                    # Perform an inside contraction
                    xcc = (1-psi)*xbar + psi*sim[-1]
                    if holdsome:
                        xcc[holdfixed] = xoriginal[holdfixed]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                        if holdsome:
                            sim[j, holdfixed] = xoriginal[holdfixed]
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim,ind,0)
        fsim = np.take(fsim,ind,0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        if disp:
            print( "Warning: Maximum number of function evaluations has "\
                  "been exceeded.")
    elif iterations >= maxiter:
        warnflag = 2
        if disp:
            print( "Warning: Maximum number of iterations has been exceeded")
    else:
        if disp:
            print( "Optimization terminated successfully.")
            print( "         Current function value: %f" % fval)
            print( "         Iterations: %d" % iterations)
            print( "         Function evaluations: %d" % fcalls[0])


    if full_output:
        retlist = x, fval, iterations, fcalls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist


def gfit(func, x0, fprime, args=(),  kwargs=dict(), maxiter=2000, ftol=0.001, factor=1., disp=False, bounds=None):
    """Perform gradient-based minimization of a user-specified function.

    :INPUTS:
      func : function
        Function that takes as input the parameters x0, optional
        additional arguments args, and optional keywords kwargs, and
        returns the metric to be minimized as a scalar.  For chi-squared
        minimization, a generalized option is :phasecurves:`errfunc`.

      x0 : sequence
        List or 1D NumPy array of initial-guess parameters, to be
        adjusted to minimize func(x0, *args, **kwargs).

      fprime : function
        Function that takes as input the parameters x0, optional
        additional arguments args, and optional keywords kwargs, and
        returns the partial derivatives of the metric to be minimized
        with regard to each element of x0. 

      args : list
        Optional arguments to func and fprime (see above)

      kwargs : dict
        Optional keywords to func and fprime (see above)

      maxiter : int
        Maximum number of iterations to run.

      ftol : scalar
        Desired tolerance on the metric to be minimized.  Iteration
        will continue until either iter>maxiter OR 
        (metric_i - metric_(i+1)) < ftol.

      factor : scalar
        Factor to scale gradient before applying each new
        iteration. Small values will lead to slower convergences;
        large values will lead to wild behavior. The code attempts to
        (crudely) tune the value of 'factor' depending on how the
        minimization process progresses.

      disp : bool
        If True, print some text to screen on each iteration.

      bounds : None, or list
        (min, max) pairs for each element in x0, defining the
        bounds on that parameter. Use None or +/-inf for one of
        min or max when there is no bound in that direction.


    :RETURNS:
      (params, metric, n_iter)

    :NOTES:
      The program attempts to be slightly clever: if the metric
      decreases by <ftol on one iteration, the code iterates one more
      time. If the termination criterion is once again met then
      minimization ends; if not, minimization continues as before.

      For quicker, smarter routines that do much the same thing, you
      may want to check out the functions in the scipy.optimize package.
    """
    # 2013-08-09 10:37 IJMC: Created
    # 2013-08-11 16:06 IJMC: Added a missing boolean flag
    
    if bounds is not None:
        bounds = np.array(bounds)

    def applyBounds(params):
        if bounds is not None:
            params = np.vstack((params, bounds[:,0])).max(0)
            params = np.vstack((params, bounds[:,1])).min(0)
        return params

    bestparams = applyBounds(x0)
    nx = bestparams.size

    metric = func(x0, *args, **kwargs)
    dmetric = 9e9
    keepFitting = True
    lastIterSaidToStop = False
    
    iter = 0
    recalcGrad = True
    if disp:
        fmtstr = '%7i   %1.'+str(np.abs(np.log(ftol)).astype(int)+2)+'f   %1.5e   %1.3e'
        print( '  ITER       METRIC         FACTOR             DMETRIC')
    while iter<maxiter and keepFitting:
        iter += 1
        if recalcGrad: grad = fprime(bestparams, *args, **kwargs)
        newparam = applyBounds(bestparams - factor * grad)
        newmetric = func(newparam, *args, **kwargs)
        if newmetric < metric:
            bestparams = newparam.copy()
            dmetric = newmetric - metric
            metric = newmetric
            if recalcGrad is True: factor *= 1.5  # we updated twice in a row!
            recalcGrad = True
            if np.abs(dmetric) < ftol:
                if disp: print( "Met termination criterion")
                if lastIterSaidToStop:
                    keepFitting = False
                else:
                    lastIterSaidToStop = True
        else:
            factor /= 2
            recalcGrad = False
            lastIterSaidToStop = False
            
        if disp: print( fmtstr % (iter, metric, factor, dmetric))

    return bestparams, metric, iter

  
def lsq(x, z, w=None, xerr=None, zerr=None, retcov=False, checkvals=True):
    """Do weighted least-squares fitting.  

    :INPUTS:
      x : sequence
        tuple of 1D vectors of equal lengths N, or the transposed
        numpy.vstack of this tuple

      z : sequence
        vector of length N; data to fit to.

      w : sequence
        Either an N-vector or NxN array of weights (e.g., 1./sigma_z**2)

      retcov : bool.  
        If True, also return covariance matrix.

      checkvals : bool
        If True, check that all values are finite values.  This is
        safer, but the array-based indexing slows down the function.

    :RETURNS: 
       the tuple of (coef, coeferrs, {cov_matrix})"""
    # 2010-01-13 18:36 IJC: Created
    # 2010-02-08 13:04 IJC: Works for lists or tuples of x
    # 2012-06-05 20:04 IJMC: Finessed the initial checking of 'x';
    #                        updated documentation, cleared namespace.
    # 2013-01-24 15:48 IJMC: Explicitly cast 'z' as type np.ndarray
    # 2014-08-28 09:17 IJMC: Added 'checkvals' option.
    # 2017-04-19 10:28 IJMC: Added option for errors in X and Y
    
    #    from numpy import vstack, dot, sqrt, isfinite,diag,ones,float, array, ndarray
    #    from numpy.linalg import pinv

    
    fitxy = False
    if xerr is None and zerr is not None:
        w = 1./np.array(zerr)**2
    elif xerr is not None and zerr is not None:
        fitxy = True
        xerr = putvecsinarray(xerr)
        zerr = putvecsinarray(zerr)


    if isinstance(x,tuple) or isinstance(x,list):
        Xmat = np.vstack(x).transpose()
    elif isinstance(x, np.ndarray) and x.ndim < 2:
        Xmat = x.reshape(len(x),1)
    else:
        Xmat = np.array(x, copy=False)

    z = np.array(z, copy=False)

    if w is None:
        w = np.diag(np.ones(Xmat.shape[0],float))
    else:
        w = np.array(w,copy=True)
        if w.ndim < 2:
            w = np.diag(w)

    if checkvals:
        goodind = np.isfinite(Xmat.sum(1))*np.isfinite(z)*np.isfinite(np.diag(w))

    nelem, nvec = Xmat.shape    
    def linear_lsq_model(p, Xmatvec):
        Xmat0 = Xmatvec.reshape(nelem, nvec)
        return np.tile(np.dot(Xmat0, p), nvec)

    if fitxy:
        # Create a model for fitting.
        lsq_model = odr.Model(linear_lsq_model)

        tileZflat = np.tile(z, nvec)
        tilegoodind = np.tile(goodind, nvec)
        etileZflat = np.tile(zerr, nvec)
        etileZflat[nelem:] *= ((etileZflat.max())*1e9)
        # Create a RealData object using our initiated data from above.
        if checkvals:
            data = odr.RealData(Xmat.flatten()[tilegoodind], tileZflat[tilegoodind], sx=xerr.flatten()[tilegoodind], sy=etileZflat[tilegoodind])
        else:
            data = odr.RealData(Xmat.flatten(), tileZflat, sx=xerr.flatten(), sy=etileZflat)

        guess, eguess = lsq(Xmat[goodind], z, checkvals=checkvals)
        
        # Set up ODR with the model and data.
        odr = odr.ODR(data, lsq_model, beta0=guess)

        # Run the regression.
        out = odr.run()
        fitcoef, covmat = out.beta, out.cov_beta

    else:
        if checkvals:
            Wmat = w[goodind][:,goodind]
            XtW = np.dot(Xmat[goodind,:].transpose(),Wmat)
            fitcoef = np.dot(np.dot(np.linalg.pinv(np.dot(XtW,Xmat[goodind,:])),XtW), z[goodind])
            covmat = (np.linalg.pinv(np.dot(XtW, Xmat[goodind,:])))
        else:
            Wmat = w
            XtW = np.dot(Xmat.transpose(),Wmat)
            fitcoef = np.dot(np.dot(np.linalg.pinv(np.dot(XtW,Xmat)),XtW), z)
            covmat = (np.linalg.pinv(np.dot(XtW, Xmat)))


    efitcoef = np.sqrt(np.diag(covmat))
        
    if retcov:
        return fitcoef, efitcoef, covmat
    else:
        return fitcoef, efitcoef


def modelspec_template(params, lam_template, template, wcoef, NPC, npix, retlam=False, verbose=False):
    """
    Return the rotationally-broadened, rv-shifited model spectrum.

     :INPUTS:
       params:      

          [0]: vsini for rotational broadening profile, in units of km/s.
               (See :func:`spec.rotationalProfile`)

          [1]: linear limb darkening coefficient for rotational
               broadening profile. (See :func:`spec.rotationalProfile`)

          [2]: rv of target divided by speed of light, (rv/c)

          [3:3+NPC]: the flux normalization
                     coefficients. These will also be passed to
                     numpy.polyval with the vector "arange(npix)/npix"

       lam_template: wavelength scale of input template. Ideally this
                       is more finely sampled than lam.

       template: template SED: the unbroadened, assumed-known spectrum
                   for this object. Ideally, this will have rather
                   broader coverage, and higher spectral resolution,
                   than the desired model.

       wcoef: array
         fitted wavelength solution.       
            
       NPC : int
         number of polynomial coefficients for continuum correction.

       npix : int
         Number of pixels in modeled spectrum.

       retlam : bool
         If True, return the tuple (model, spectrum).

       :EXAMPLE:
         ::

            XXX update this for vsini/LLD case!!!

            NPW = 3
            npix = wobs.size
            pix = np.arange(npix, dtype=float)/npix
            ccoef = [1./np.median(template)]
            NPC = len(ccoef)
            guess = np.concatenate(([17, 1e-4, 9, 1], ccoef))

            mygmod, mygw = fit_atmo.modelspec_tel_template(guess, lam_template, template, lam_atmo, atmo, NPC, npix, retlam=True)

       Things like flux conservation and line-broadening are not
       well-treated in this function!
       """
    # 2013-05-08 07:38 IJMC: Created
    # 2013-08-06 10:28 IJMC: Updated to use vsini and LLD
    # TODO: example says fit_atmo.modelspec_tel_template?
    # 2022-11-29 XQ: remove wcoef as parameter

    vsini = params[0]
    lld = params[1]
    rv = params[2]

    if vsini<0:
        vsini = 0

    continuum_coefs = params[3:3+NPC]
    
    lam_template = np.array(lam_template, copy=False)
    template = np.array(template, copy=False)

    if lam_template.shape!=template.shape or lam_template.ndim!=1:
        return -1
        
    pix = np.arange(npix, dtype=float)/npix

    # Create model-convolution Kernel and convolve template:
    pixsize_ms = np.diff(lam_template).mean()/lam_template.mean() * const.c
    xkern = np.arange(-int(1200.*vsini/pixsize_ms), int(1200.*vsini/pixsize_ms)+1)
    if xkern.size>=template.size:
        xkern = np.arange(-template.size/2, template.size/2)
    
    if verbose:
        print("xkern>>", xkern)

    if xkern.size<=1:
        rotational_profile = np.array([1])
    else:
        dv = xkern * pixsize_ms
        rotational_profile = rotationalProfile([vsini*1000., lld, 0], dv)
        if verbose:
            print("rotational_profile>>", rotational_profile)
        rotational_profile /= rotational_profile.sum()
        #kern = gaussian([1., fwhm/2.3548, 0, 0], xkern)
        #kern /= kern.sum()

    new_template = np.convolve(template, rotational_profile, 'same')

    # Shift template to specified RV & interpolate to wavelength grid
    lam = np.polyval(wcoef, pix)
    output = np.interp(lam, lam_template*(1.+rv), new_template, left=0., right=0.)

    # Multiply by appropriate normalization polynomial
    output *= np.polyval(continuum_coefs, pix)

    if retlam:
        ret = (output, lam)
    else: 
        ret = output
        
    return ret

def modelspec_tel_template(params, lam_template, template, lam_atmo, atmo, wcoef, NPC, npix, retlam=False, verbose=False):
    """
    Return the rotationally-broadened, rv-shifited, telluric-corrected model spectrum.

     :INPUTS:
       params:      

          [0]: vsini for rotational broadening profile, in units of km/s.
               (See :func:`spec.rotationalProfile`)

          [1]: linear limb darkening coefficient for rotational
               broadening profile. (See :func:`spec.rotationalProfile`)

          [2]: rv of target divided by speed of light, (rv/c)

          [3]: fwhm for telluric convolution, in units of PIXELS of
               lam_atmo. (This may not be ideal if your wavelength
               scale in 'lam_atmo' changes rapidly).

          [4]: effective scaling of telluric transmission specturm in
              'atmo'.  This will be scaled as: 1.0 - (scale*(1.0 -
              atmo)). After convolution, any negative values will be
              set to zero. Thus line-broadening is explicitly *not*
              treated correctly.

          [5:5+NPC]: the flux normalization
                     coefficients. These will also be passed to
                     numpy.polyval with the vector "arange(npix)/npix"

       lam_template: wavelength scale of input template. Ideally this
                       is more finely sampled than lam.

       template: template SED: the unbroadened, assumed-known spectrum
                   for this object. Ideally, this will have rather
                   broader coverage, and higher spectral resolution,
                   than the desired model.

       lam_atmo: wavelength scale of input telluric transmission
                       spectrum. Ideally this is more finely sampled
                       than the desired output wavelength scale.

       atmo: template of telluric transmission. Ideally, this will
                   have higher spectral resolution than the spectrum
                   you wish to model.

       wcoef: array
         fitted wavelength solution.  

       NPC : int
         number of polynomial coefficients for continuum correction.

       npix : int
         Number of pixels in modeled spectrum.

       retlam : bool
         If True, return the tuple (model, spectrum).

       :EXAMPLE:
         ::

            XXX update this for vsini/LLD case!!!

            NPW = 3
            npix = wobs.size
            pix = np.arange(npix, dtype=float)/npix
            wcoef = np.polyfit(pix, wobs, NPW-1)
            ccoef = [1./np.median(template)]
            NPC = len(ccoef)
            guess = np.concatenate(([17, 1e-4, 9, 1], wcoef, ccoef))

            mygmod, mygw = fit_atmo.modelspec_tel_template(guess, lam_template, template, lam_atmo, atmo, NPW, NPC, npix, retlam=True)

       Things like flux conservation and line-broadening are not
       well-treated in this function!
       """
    # 2013-05-08 07:38 IJMC: Created
    # 2013-08-06 10:28 IJMC: Updated to use vsini and LLD
    import pdb
    #import spec

    vsini = params[0]
    lld = params[1]
    rv = params[2]
    atmo_fwhm = params[3]
    if params[4]<0:
        atmo_scale = 0
    else:
        atmo_scale = params[4]

    if vsini<0:
        vsini = 0

    continuum_coefs = params[5:5+NPC]
    
    lam_template = np.array(lam_template, copy=False)
    template = np.array(template, copy=False)
    lam_atmo = np.array(lam_atmo, copy=False)
    atmo = np.array(atmo, copy=False)

    if lam_template.shape!=template.shape or lam_template.ndim!=1:
        return -1
    if lam_atmo.shape!=atmo.shape:
        return -1
        
    pix = np.arange(npix, dtype=float)/npix

    # Create model-convolution Kernel and convolve template:
    pixsize_ms = np.diff(lam_template).mean()/lam_template.mean() * const.c
    xkern = np.arange(-int(1200.*vsini/pixsize_ms), int(1200.*vsini/pixsize_ms)+1)

    if xkern.size>=template.size:
        xkern = np.arange(-template.size/2, template.size/2)

    if verbose:
        print("xkern>>", xkern)
    if xkern.size<=1:
        rotational_profile = np.array([1])
    else:
        dv = xkern * pixsize_ms
        rotational_profile = rotationalProfile([vsini*1000., lld, 0], dv)
        if verbose:
            print("rotational_profile>>", rotational_profile)
        rotational_profile /= rotational_profile.sum()
        #kern = gaussian([1., fwhm/2.3548, 0, 0], xkern)
        #kern /= kern.sum()

    new_template = np.convolve(template, rotational_profile, 'same')

    # Create telluric-convolution Kernel and convolve scaled telluric spectrum:
    new_atmo = 1.0 - atmo_scale*(1.0 - atmo)
    xkern = np.arange(int(-5*atmo_fwhm), int(5*atmo_fwhm))
    if xkern.size>=new_atmo.size:
        xkern = np.arange(-new_atmo.size/2, new_atmo.size/2)

    if xkern.size<=1:
        pass
    else:
        kern = gaussian([1., atmo_fwhm/2.3548, 0, 0], xkern)
        kern /= kern.sum()
        new_atmo = np.convolve(new_atmo, kern, 'same')

    new_atmo[new_atmo<0] = 0.
    if verbose:
        print("new_atmo>>", new_atmo)

    # Shift template to specified RV & interpolate to wavelength grid
    #pdb.set_trace()
    lam = np.polyval(wcoef, pix)
    if verbose:
        print("lam>>", lam)
        print("wcoef>>", wcoef)
        print("lam_template>>", lam_template)
        print("rv>>", rv)
        print("new_template>>", new_template)
    new_template = np.interp(lam, lam_template*(1.+rv), new_template, left=0., right=0.)
    if verbose:
        print("new_template>>", new_template)
    output = new_template * np.interp(lam, lam_atmo, new_atmo, left=0., right=0.)

    # Multiply by appropriate normalization polynomial
    output *= np.polyval(continuum_coefs, pix)


    if retlam:
        ret = (output, lam)
    else: 
        ret = output
        
    return ret

def rotationalProfile(delta_epsilon, delta_lam):
    """Compute the rotational profile of a star, assuming solid-body
    rotation and linear limb darkening.

    This uses Eq. 18.14 of Gray's Photospheres, 2005, 3rd Edition.

    :INPUTS:

      delta_epsilon : 2-sequence

        [0] : delta_Lambda_L = lambda * V * sin(i)/c; the rotational
              displacement at the stellar limb.

        [1] : epsilon, the linear limb darkening coefficient, used in
              the relation I(theta) = I0 + epsilon * (cos(theta) - 1).

        [2] : OPTIONAL! The central location of the profile (otherwise
              assumed to be located at delta_lam=0).

      delta_lam : scalar or sequence
        Wavelength minus offset: Lambda minus lambda_0.  Grid upon
        which computations will be done.

    :EXAMPLE:
      ::

        import pylab as py
        import spec

        dlam = py.np.linspace(-2, 2, 200) # Create wavelength grid
        profile = spec.rotationalProfile([1, 0.6], dlam)

        py.figure()
        py.plot(dlam, profile)
    """
    # 2013-05-26 10:37 IJMC: Created.

    delta_lambda_L, epsilon = delta_epsilon[0:2]
    if len(delta_epsilon)>2:  # optional lambda_offset
        lamdel2 = 1. - ((delta_lam - delta_epsilon[2])/delta_lambda_L)**2
    else:
        lamdel2 = 1. - (delta_lam/delta_lambda_L)**2
    
    if not hasattr(delta_lam, '__iter__'):
        delta_lam = np.array([delta_lam])

    ret = (4*(1.-epsilon) * np.sqrt(lamdel2) + np.pi*epsilon*lamdel2) / \
        (2*np.pi * delta_lambda_L * (1. - epsilon/3.))    

    ret[lamdel2<0] = 0.

    return ret


def gaussian(p, x):
    """ Compute a gaussian distribution at the points x.

        p is a three- or four-component array, list, or tuple:

        y =  [p3 +] p0/(p1*sqrt(2pi)) * exp(-(x-p2)**2 / (2*p1**2))

        p[0] -- Area of the gaussian
        p[1] -- one-sigma dispersion
        p[2] -- central offset (mean location)
        p[3] -- optional constant, vertical offset

        NOTE: FWHM = 2*sqrt(2*ln(2)) * p1  ~ 2.3548*p1

        SEE ALSO:  egaussian"""
    #2008-09-11 15:11 IJC: Created for LINEPROFILE
    # 2011-11-10 12:00 IJMC: Don't copy x or p inputs; fiddled with
    # background addition and ordering of operations
    # 2012-12-23 12:12 IJMC: Made it a tad faster.
    
    if not isinstance(x, np.ndarray):
        x = array(x, dtype=float, copy=False)

    y = p[0]/(p[1]*sqrt(2*np.pi)) * exp(-0.5 * ((x-p[2])/p[1])**2)

    if len(p)>3:
        y += p[3]
    
    return y
