import numpy as np
import matplotlib.pyplot as plt


def make_deltaspec(loc, ew, win, **kw):
    """
    Create a delta-function line spectrum based on a wavelength grid
    and a list of line locations and equivalent widths.

    :INPUTS:
       loc -- location of lines in the emission frame of reference

       ew  -- equivalent widths of lines, in units of wavelength grid.
               Positive values are emission lines.

       win -- wavelength grid in the emission frame, with values
              monotonically increasing (best if it is linearly spaced)

       All inputs should be lists or one-dimensional arrays of scalars

    :OPTIONAL_INPUTS:
       cont=None -- set continuum values in the emission frame;

       nearest=False  -- if True, use full pixels instead of partial

       verbose=False  -- if True, print out various messages

    :OUTPUTS:
      s  -- delta-function line spectrum, with a continuum level of zero
    
    :EXAMPLE: (NEEDS TO BE UPDATED!):
       ::

          w   = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
          loc = [2.1, 2.35, 2.62]
          ew  = [0.1, .02, .01]
          s = linespec(loc, ew, w)
          print s  #  --->  [0, 1, 0, 0.1, 0.1, 0, 0.08, 0.02]

    :NOTE:  This may give incorrect results for saturated lines.
    """
    # 2008-12-05 13:31 IJC: Created
    # 2008-12-10 13:30 IJC: Added continuum option, reworked code some.
    # 2008-12-12 12:33 IJC: Removed RV option

    # Check inputs
    loc = np.array(loc).copy().ravel()
    ew  = np.array(ew ).copy().ravel()
    win = np.array(win).copy().ravel()

    defaults = dict(cont=None, nearest=False, verbose=False)
    for key in defaults:
        if (not key in kw):
            kw[key] = defaults[key]
    verbose = bool(kw['verbose'])
    nearest = bool(kw['nearest'])
    contset = kw['cont']!=None

    if contset.all():
        cont = np.array(kw['cont']).copy()
        if len(cont)!=len(win):
            print( "Wavelength grid and continuum must have the same length!")
            return -1
    else:
        cont = np.ones(win.shape)

    nlines = len(loc)
    if nlines != len(ew):
        if verbose:  print( "len(loc)>>" + str(len(loc)))
        if verbose:  print( "len(ew)>>" + str(len(ew)))
        print( "Line locations and equivalent widths must have same length!")
        return -1

    # Only use lines in the proper wavelength range
    nlineinit = len(loc)
    lind = (loc>=win.min()) * (loc<=win.max())
    loc = loc[lind]
    ew  =  ew[lind]
    nlines = len(loc)

    s = cont.copy()
    d = np.diff(win).mean()

    if verbose:  print( "s>>" + str(s))

    for ii in range(nlines):
        lineloc = loc[ii]
        lineew  = ew[ii]
        index = (win<lineloc).sum() - 1
        if nearest:
            s[index+1] = s[index]-cont[index]*lineew/d
        elif index==len(win):
            s[index] = s[index] - cont[index]*lineew/d
        else:
            s[index] = s[index] - lineew*cont[index]* \
                (win[index+1] - lineloc)/d/d
            s[index+1] = s[index+1] - lineew*cont[index+1] * \
                (lineloc - win[index])/d/d
        
        if verbose:  
            print( "(lineloc, lineew)>>" + str((lineloc, lineew)))
            print( "(index, d)>>" + str((index,d)))

    if verbose:
        print( "(nlineinit, nline)>>" + str((nlineinit, nlines)))
    return s

def dsa(r, i, Nk, **kw):
    """
    Computational tool for Difference Spectral Analysis (DSA)
    
    :INPUTS:
       R -- reference spectrum.  This should have the highest possible
            signal-to-noise and the highest spectral resolution.
       I -- Current spectrum to be analysed.
       Nk -- number of pixels in the desired convolution kernel

    :OPTIONS:
       w       -- weights of the pixel values in I; typically (sigma)^-2
           (Not HANDELED CORRECTLY?!?!?)
       noback  -- do not fit for a variable background; assume constant.
       tol=1e-10 -- if matrix determinant is less than tol, use
                    pseudoinverse rather than straight matrix
                    inversion
       verbose -- Print output statements and make a plot or two
       retinv -- return a fourth output, the Least Squares inverse
                 matrix (False by default)

    :OUTPUTS:       (M, K, B, C):
       M -- R, convolved to match I
       K -- kernel used in convolution
       B -- background offset
       C -- chisquared of fit. If no weights were specified, weights
            are set to unity for this calculation.

    :OPTIONS:
       I -- inverse matrix

    :NOTES:
        Best results are obtained with proper registration of the spectra.
        Also, beware of edge effects.  As a general rule, anything within
        a kernel width of the edges is suspect.
        Also

    :SEE_ALSO:  
       :func:`dsamulti`
    
    Based on the 2D Bramich (2008) DIA algorithm
    -----
    2008-11-14 10:56 IJC: Created @ UCLA.
    2008-11-18 11:12 IJC: Registration now works correctly
    2008-12-09 16:10 IJC: Somewhat optimized
    2009-02-26 22:06 IJC: Added retinv, changed optional input format
    """

    defaults = dict(verbose=False, w=None, noback=False, tol=1e-10, retinv=False)
    for key in defaults:
        if (not (key in kw)):
            kw[key] = defaults[key]
    verbose = bool(kw['verbose'])
    noback = bool(kw['noback'])
    retinv = bool(kw['retinv'])
    w = kw['w']
    if verbose:
        print( "kw>>" + str(kw))

    if noback:
        if verbose: print( "Not fitting for a variable background...")

    tol = 1e-10  # tolerance for singularity

    r = np.array(r, copy=True)
    i = np.array(i, copy=True)
    Nk = int(Nk)  # length of kernel
    dx = int(np.floor(Nk/2))

    if w==None:
        w = np.ones(len(r), dtype=float)

    Nr = len(r)  # length of Referene
    ind = np.arange(Nr-Nk+1, dtype=int)
    wind = w[ind]
        
    if noback:    
        U = np.zeros((Nk,Nk), dtype=float)
        b = np.zeros(Nk, dtype=float)
    else:
        U = np.zeros((Nk+1,Nk+1), dtype=float)
        b = np.zeros(Nk+1, dtype=float)

    # Build the b vector and U matrix
    tempval0 = w[ind+dx] * i[ind+dx]
    for p in range(Nk):
        b[p] = (tempval0 * r[ind+p]).sum()
        tempval2 = wind*r[ind+p]
        for q in range(p, Nk):
            U[p,q] = (tempval2 * r[ind+q]).sum()
            U[q,p] = U[p,q]

    if not noback:
        b[Nk] = (w[ind+dx] * i[ind+dx]).sum()
        for q in range(Nk):
            U[Nk, q] = (wind * r[ind+q]).sum()
            U[q, Nk] = U[Nk, q]

        U[Nk,Nk] = wind.sum()
    
    detU = np.linalg.det(U)
    if verbose: print( "det(U) is:  " + str(detU))

    if detU<tol:
        print( "Singular matrix: det(U) < tol.  Using pseudoinverse...")
        if verbose: 
            print( 'U>>',U)
        invmat = np.linalg.pinv(U)
    else:
        invmat = np.linalg.inv(U)

    a = np.dot(invmat, b)

    if noback:
        K = a
        B0 = 0.0
    else:
        K = a[0:len(a)-1]
        B0 = a[-1]

    m = deconvolve1d(r, K) + B0

    chisq  = (wind * (i[ind] - m[ind])**2).sum()

    if verbose:
        chisq0 = ( wind * (i[ind] - r[ind])**2 ).sum()
        print("Background: " + str(B0))
        print("For the (" + str(Nr) + " - " + str(Nk+1) + ") = " + str(Nr-Nk-1) + " DOF:")
        print("Red. Chisquared (I-R): " + str(chisq0/(Nr-Nk-1)))
        print("Red. Chisquared (I-M): " + str(chisq/(Nr-Nk-1)))
    
        plt.figure(); plt.subplot(311)
        plt.plot(r, '--'); plt.plot(i, '-x'); plt.plot(m, '-..'); plt.legend('RIM'); 
        plt.subplot(312); plt.plot(r - i, '--'); plt.plot(m - i); plt.legend(['R-I', 'M-I'])
        plt.subplot(313); plt.plot(K, '-o'); plt.grid('on'); plt.legend(['Kernel']); 

    if retinv:
        return (m, K, B0, chisq, invmat)
    else:
        return (m, K, B0, chisq)

def deconvolve1d(a, b, extend='nearest'):
    """
    Compute a 1D deconvolution in the style of Bramich 2008.

    :INPUTS:
        'a' should be longer than 'b' -- i.e., 'b' is the kernel.
        'extend' tells how to extend the boundaries -- either
        'nearest'-neighbor or a number

    :NOTES:
      This is "reversed" from the canonical definition of the convolution.

    :SEE_ALSO:   
      :func:`dsa`
    """
    # 2008-11-14 18:26 IJC: Created
    na = len(a)
    nb = len(b)
    n = max(na, nb)
    
    dx = int(np.floor(nb/2))

    if extend=='nearest':
        X = a[-1]
    else:
        X = extend

    a2 = X + np.zeros(na+nb-1, dtype=float)
    a2[dx:dx+na] = a
    
    bmat = np.tile(b, (n,1))
    amat = np.zeros((n, nb), dtype='float')
    for ii in range(na):
        amat[ii,:] = a2[range(ii,ii+nb)]

    c = np.sum(amat * bmat, axis=1)
        
    return c

def dao_getlines(f_linelist):
    """
    Read the line locations and equivalent widths from a DAOSPEC output file.

    Example:
      f_linelist = 'model_spec.clines'
      (lineloc, lineew, linespec) = getlines(f_linelist)
    """
    #2009-02-22 10:15 IJC: Initiated

    # Get the line locations and EWs:
    with open(f_linelist, 'r') as f:
        raw = f.readlines()

    dat = np.zeros([len(raw), 2], dtype=float)                                                 
    for i, line in enumerate(raw):                                         
        dat[i,:]= list(map(float, line.split()[0:2]))

    lineloc = dat[:,0]
    lineew = dat[:,1]/1e3
    linespec = [line.split()[-1] for line in raw]
    return (lineloc, lineew, linespec)
