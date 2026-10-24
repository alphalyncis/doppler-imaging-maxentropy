# 2014-01-23 14:51 IJMC: Playground to investigate feasibility of
#                        Doppler Imaging.



from pylab import *
import os
import dia, dao, pyfits
from scipy import interpolate
from scipy import signal
import nsdata as ns
import analysis as an
import numpy as np
import fit_atmo
from scipy import signal
import tools

_model = './'
pmod = 'output'
#npix = wobs.shape[1]

dloglam = .02 # 50./2300.
npix = 2048
vsini = 20. # km/s
LLD = 0.5 # linear limb darkening coefficient
RV = 0./3e5  # km/s, target relative to telluric
atmo_fwhm_pix = 5. # pixels; fix this!
resolution = 5e4 # spectral resolution
lambda_factor = 0.004 # lambda^2 / (R * d_{lam_FSR})
telluric_scaling = 1.
NPW = 2
NPC = 1

pix_arcsec = 0.1 # arcsec per pixel
slitwidth = 0.2 # arcsec
pix_sr = (0.1 / 206265)**2 # steradian
total_efficiency = 0.1
readnoise = 10. # electrons
itime = 1200 # sec
nrows = 10 # detector rows
diameter = 820. #  cm, diameter
distance = 2.2  # parsecs
radius = 1  # Jupiter radii


minlam = 0.785
maxlam = 5.4
#minlam, maxlam = 1.2, 1.3
area = np.pi * (diameter/2.)**2 # cm^2

plotalot=True
fs = 16

nbins = np.ceil(np.log10(maxlam/minlam) / np.log10(1. + dloglam)).astype(int)
lam_centers = minlam * (1. + dloglam)**np.arange(nbins)

#lam_lims = np.zeros((0, 2), dtype=float)
#while lam_lims.size==0 or lam_lims.max()<maxlam:
#    if lam_lims.size==0:
#        thislam = minlam
#    else:
#        thislam = lam_lims.max()
#    newlims = [thislam, thislam + thislam**2/(resolution * lambda_factor)]
#    lam_lims = np.vstack((lam_lims, newlims))
#
#nbins = lam_lims.shape[0]
#lam_centers = lam_lims.mean(1)
_mod = os.path.expanduser('~') + '/proj/pcsa/data/model/'
modelfn =  _mod + '/lte016-4.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte023-4.5-0.5a+0.2.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte015.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte011-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte008-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte020-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte012-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte018-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte004.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte006.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte013-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte014.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte013.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte009-5.0-0.0a+0.0.BT-Settl.spec.7.fits'
modelfn =  _mod + '/lte004.5-5.0-0.0a+0.0.BT-Settl.spec.7.fits'



telfn = '/Users/ianc/proj/pcsa/data/atmo/transdata_0,5-5_mic.fits' #transdata_2,28-2,37_mic_hires.fits'
#telradfn = '/Users/ianc/proj/pcsa/data/atmo/raddata_1,5-5,5_losres_erg-s-cm2-um-sr.fits'
telradfn = '/Users/ianc/proj/pcsa/data/atmo/mauna_kea_emission_h2o=16_airmass=1,5_photon-sec-arcsec2-nm-m2.fits'
atm0 = pyfits.getdata(telfn).T
radatm0 = pyfits.getdata(telradfn)


model = pyfits.getdata(modelfn) # erg/s/cm2/micron/sr


band_inds = [0.8, 1.11], [1.16, 1.35], [1.5, 1.79], [2, 2.47], [3.4, 4.1]
band_names = ['Z/Y', 'J', "H", "K", 'L']
if False:
    npixes = [(4*np.diff(el)/np.mean(el)) / (2. / resolution) for el in band_inds]
    lam_centers = [np.mean(el) for el in band_inds]
    dloglams = [np.diff(el)/np.mean(el) for el in band_inds]



for distance in [2.2]: #, 4, 8, 16]:
    geometric_scalefactor = np.pi * ((radius * an.rjup)/(distance * an.pc))**2 # sr
    ew = np.zeros(nbins)
    ew_dao = np.zeros(nbins)
    median_flux = np.zeros(nbins)
    median_atmo = np.zeros(nbins)
    median_skyflux = np.zeros(nbins)
    median_snr = np.zeros(nbins)



    #for jj, dloglam, npix in zip(range(5), dloglams, npixes):
    for jj in xrange(nbins):
        #lolim, hilim = lam_lims[jj]
        lolim = lam_centers[jj] * (1. - dloglam/2.)
        hilim = lam_centers[jj] * (1. + dloglam/2.)

        tind = (model[0]>lolim) * (model[0] < hilim)
        lam_template = model[0,tind]
        template = model[1,tind]

        #template /= np.median(template)
        aind = (atm0[0]>lolim) * (atm0[0] < hilim)
        true_atmo = atm0[1,aind]
        true_lam_atmo = atm0[0,aind]
        lam_atmo = lam_template.copy()
        atmo = np.ones(lam_atmo.size)

        lam_poly = [(hilim-lolim), lolim]
        params = [vsini, LLD, RV, 3e5/resolution, telluric_scaling] + lam_poly + [1]
        observation, lam = fit_atmo.modelspec_tel_template2(params, lam_template, template, lam_atmo, atmo, NPW, NPC, npix, retlam=True)
        telluric_observation, tel_lam = fit_atmo.modelspec_tel_template2(params, lam_template, template, true_lam_atmo, true_atmo, NPW, NPC, npix, retlam=True)

        goodind = ((observation>0).nonzero()[0][0], -(observation[::-1]>0).nonzero()[0][0]-1)
        lam = lam[goodind[0]:goodind[-1]]
        observation = observation[goodind[0]:goodind[-1]] # erg/s/cm2/micron
        telluric_observation = telluric_observation[goodind[0]:goodind[-1]] # erg/s/cm2/micron
        dlam = concatenate([[0], diff(lam)])

        photon_energy = an.h * an.c/(lam * 1e-6) * 1e7# erg / photon
        telluric_observation *= (geometric_scalefactor / photon_energy) # photon/s/cm2/micron
        # phot/sec/arcsec2/nm/m2'
        lo_background_flam = np.interp(lam, radatm0[0], radatm0[1]) # 
        lo_background = lo_background_flam * slitwidth * pix_arcsec * dlam * 1000/ 1e4 # phot/sec/pixel/cm2

        integration = telluric_observation * itime * area * total_efficiency * dlam # photons per detector column
        sky_flux = lo_background * itime * area # photons per detector column
        snr = integration / np.sqrt(integration + sky_flux + (readnoise*nrows)**2)

        if False:
            pmod = os.path.split(modelfn)[1].replace('BT-Settl.spec.7.fits', '_dao' ).replace('.',',') 
            dao.runspec(lam*1e4, observation, pmod, [lam.min()*1e4 +0.01, lam.max()*1e4 - 0.01], 8, order=3)
            #dao.runspec(template*1e4, observation, pmod, [lam.min()*1e4, lam_template.max()*1e4], 8, order=3)
            print "You should really examine those automatically-measured line strengths by hand...!"
            #f_linelist   = _mod + pmod + '.clineslsd'
            f_linelist   = pmod + '.clineslsd'
            f_pspec      = pmod + '.fits'
            f_pspec_cont = pmod + 'C.fits'

            (lineloc, lineew, linespec) = dao.getlines(f_linelist)  
            pspec_cont = pyfits.getdata(f_pspec_cont) 
            hdr_pspec_cont = pyfits.getheader(f_pspec_cont) 
            wspec = hdr_pspec_cont['crval1'] + np.arange(pspec_cont.size)*hdr_pspec_cont['cdelt1']
            #spline = interpolate.UnivariateSpline(wspec, pspec_cont, s=0.0, k=1.0)
            deltaspec0 = ns.linespec(lineloc, lineew, wspec, verbose=False, cont=pspec_cont)
            m0,k0,b0,c0 = dia.dsa(deltaspec0, observation, 123)



        xpix = np.linspace(0, 1, lam.size)
        cfit = an.polyfitr(lam, observation, 2, 0.5, clip='below', fev=lam.size*0.8)
        pspec_cont = np.polyval(cfit, lam)
        pspec_cont[pspec_cont<np.sort(observation)[int(lam.size*0.01)]] = observation[pspec_cont<np.sort(observation)[int(lam.size*0.01)]]
        pspec_cont[pspec_cont<observation] = observation[pspec_cont<observation]
        pspec_cont = signal.medfilt(pspec_cont, 101)

        dlam = concatenate([[0], diff(lam)])
        ew[jj] = (1e4*dlam * (1. - observation / pspec_cont)).sum()
        median_flux[jj] = np.median(integration)
        median_skyflux[jj] = np.median(sky_flux)
        median_snr[jj] = np.median(snr)
        median_atmo[jj] = np.median(true_atmo)

        if plotalot:
            figure()
            ax1=subplot(211, position=[.1, .4, .85, .5])
            plot(lam, median_flux[jj]*observation, '-k')
            xlabel('Wavelength ($\mu$m', fontsize=fs)
            ylabel('Emergent Flux Density', fontsize=fs)
            ax1.set_xticklabels([])
            yticks(ax1.get_yticks()[1:])

            ax2=subplot(212, position=[.1, .1, .85, .3])
            plot(true_lam_atmo, true_atmo, '-', color='gray')
            ylim(0,1.1)
            ylabel('Telluric Transmission', color='k', fontsize=fs)
            [ax.set_xlim(lam.min()+0.001, lam.max()-0.001) for ax in [ax1, ax2]]
            [ax.minorticks_on() for ax in [ax1, ax2]]

        #ew_dao[jj] = lineew.sum()
        #p = os.popen('ls %s*' % pmod )
        #filenames = p.readlines()
        #p.close()
        #for filename in filenames:
        #    os.rename(filename.strip(), filename.replace('_dao', '_%i-%i_dao' % (lolim*1e4, hilim*1e4)).strip())


    metric = median_snr * ew / 15000. / 1.7


    xt = [0.6, 0.8, 1, 2, 3, 4, 5]
    names = 'metric', 'median_snr', 'ew', 'median_flux', 'median_atmo'
    dy = .8/len(names)

    fig = figure(-1)
    fig.text(0.5, 0.93, os.path.split(modelfn)[-1], fontsize=fs, horizontalalignment='center')
    for ii, name in enumerate(names):
        ax = subplot(111, position=[.15, .1+ii*dy, .8, dy*0.999])
        exec("semilogx(lam_centers, %s, '-k', drawstyle='steps-mid', linewidth=2)" % name)
        xticks(xt, xt)
        xlim(min(xt), max(xt))
        ylabel(name)
        if ii>0:  
            ax.set_xticklabels([])
        else:
            xlabel('Wavelength ($\mu$m)', fontsize=fs)


    names = 'metric', 'ew', 'median_snr', 
    ylabs = 'Mapping\nSensitivity', '$\Sigma$ EW $(\AA)$', 'S/N', 
    lims = [0, 30] , [0, 350], [0, 950],
    lims = [0, 1.1], [0, 350], [0, 300],
    dy = .8/len(names)

    fig = figure(-2, [4.5, 6])
    #fig.text(0.5, 0.93, os.path.split(modelfn)[-1], fontsize=fs, horizontalalignment='center')
    axs = []
    for ii, name in enumerate(names):
        ax = subplot(111, position=[.2, .1+ii*dy, .75, dy*0.999])
        axs.append(ax)
        exec("semilogx(lam_centers, %s, '-k', drawstyle='steps-mid', linewidth=2)" % name)
        xticks(xt, xt)
        xlim(min(xt), max(xt))
        ylabel(ylabs[ii], fontsize=fs)
        ylim(lims[ii])
        minorticks_on()
        if ii>0:  
            ax.set_xticklabels([])
            yticks(ax.get_yticks()[1:])
        else:
            xlabel('Wavelength ($\mu$m)', fontsize=fs)
        if ii<(len(names))-1:  yticks(ax.get_yticks()[0:-1])
        text(.04, .08, '(%s)' % 'abcde'[len(names)-ii-1], fontsize=fs, transform=ax.transAxes)

    for ax in axs:
        for bi in band_inds:
            if min(bi)>1:
                tools.drawRectangle(min(bi), min(ax.get_ylim()), np.diff(bi), np.diff(ax.get_ylim()), ax=ax, color="0.7")


    blines = ''
    band_integrated_metrics = []
    for bi, bn in zip(band_inds, band_names):
        ind = (lam_centers > bi[0]) * (lam_centers <= bi[1])
        val = np.sqrt((metric[ind]**2).sum())
        blines += ('%04s band: %1.2f integrated metric\n' % (bn, val))
        band_integrated_metrics.append(val)

    blines += 'Diameter: %1.1f m\nDistance: %1.1f pc\n' % (diameter/100., distance)
    fig = figure(tools.nextfig(), [4.5, 6])
    text(0.5, 0.5, blines.replace('\n', '\n\n'), horizontalalignment='center', verticalalignment='center', fontsize=fs)


    ##################################################
    # Save everything
    ##################################################

    temp0 = os.path.split(modelfn)[1]
    teff = float(temp0[4:temp0.find('-')]) * 100
    printfn = ('DI_%iK_%1.1fpc_%1.1fm_summary' % (teff, distance, diameter/100.)).replace('.', ',')

    save_objects = 'dloglam', 'npix', 'vsini', 'resolution', 'pix_arcsec', 'slitwidth', 'total_efficiency', 'readnoise', 'itime', 'nrows', 'diameter', 'distance', 'radius', 'modelfn', 'ew', 'lam_centers', 'metric', 'median_snr', 'ew', 'median_flux', 'median_atmo', 'band_inds', 'band_names', 'band_integrated_metrics'
    save = dict()
    for name in save_objects:
        try:
            exec ("save[name] = %s" % name)
        except:
            print "Could not save field: %s" % name
            save[name] = None

    picklefn = printfn + '_output.pickle'
    pickleiter = 1
    while os.path.isfile(picklefn):
        picklefn = printfn + '_output_%i.pickle' % pickleiter
        pickleiter += 1


    tools.savepickle(save, picklefn)
    tools.printfigs(picklefn.replace('pickle','pdf'))
    close('all')



#ylabel('


#out = spec.runlblrtm([1.5, 6], alt=3, dotrans=True, dorad=True, res=200)
