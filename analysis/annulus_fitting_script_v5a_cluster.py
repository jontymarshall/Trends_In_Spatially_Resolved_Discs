#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:07 2019

@author: jonty
"""
direc = '/home/jmarshall/ring_model/'
#Procedure to test model ring creation, rotation, resampling, and convolution routine

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from scipy.ndimage.interpolation import rotate 
from scipy.ndimage.interpolation import shift
from scipy import interpolate
from photutils.centroids import fit_2dgaussian
from skimage.transform import rescale
import json
import emcee
import corner
import dill
import pathos.multiprocessing as mp
import time
from skimage.measure import block_reduce


def make_disc(data,theta,fstar,pixscale=1.6):
    try:
        f,r,d,i,p,dx,dy = theta
    except:
        f = theta[0]
        r = theta[1]
        d = theta[2]
        i = theta[3]
        p = theta[4]
        dx = theta[5]
        dy = theta[6]
    #print(f,r,d,i,p,dx,dy)
    #stellar parameters
    #lstar = data["main_results"][0]["lstar"]
    dstar = 1./data["main_results"][0]["plx_arcsec"]
    #tstar = data["main_results"][0]["Teff"]
    
    #test disc parameters
    radius = 10**r  # au - radius of peak disc flux in debris belt
    dr     = d  # fractional width dR/R of debris belt
    incl   = (180./np.pi)*np.arccos(i)  # degrees
    posang = (180./np.pi)*np.arccos(p)  # degrees + 90 because rotation starts at positive x axis
    fscale = 10**f  # mJy - scaling factor for disc - total flux
    
    rwidth  = dr*radius # au - width of debris belt (Gaussian sigma)
    rsigma  = rwidth/(2*(2*np.log(2))**0.5)
    
    scaleheight = 0.05 # open angle for the disc
    
    rscale = 5.0 # sample distance between star and disc with npts elements
    if radius > 200. :
        rscale = radius/20.
    if radius < 50. :
        rscale = radius/20.
    #calculate model grid for dust density calculation    
    nel = int(2*(radius + 5*rsigma)/rscale) #adjust model size depending on radius and width of disc - must be odd!
    #print(nel)
    if (nel % 2) == 0:
        nel = int(nel + 1)
    
    #create grid, reorient based on offsets, posa and incl using transformation and rotation from Zodipic 2.1
    nc = int((nel-1)/2)
    xg = rscale*(np.arange(nel) - nc)
    yg = rscale*(np.arange(nel) - nc)
    zg = rscale*(np.arange(nel) - nc)
    xx,yy,zz = np.meshgrid(xg, yg, zg) 
    
    #Offsets of ring in x,y,z axes
    deltax = dx*dstar
    deltay = dy*dstar
    deltaz = 0.0
    
    #Angles for transformation
    c0 = np.cos(posang*np.pi/180.0)
    c1 = np.cos(incl*np.pi/180.0)
    c2 = np.cos(0.5*np.pi)
    s0 = np.sin(posang*np.pi/180.0 )
    s1 = np.sin(incl*np.pi/180.0)
    s2 = np.sin(0.5*np.pi)
    
    #Transformations
    trans1 = -(s0*c1*c2 + c0*s2)*yy + s1*c2*zz + c2*deltax -s2*deltay
    trans2 = -(s0*c1*s2 - c0*c2)*yy + s1*s2*zz + s2*deltax +c2*deltay
    trans3 = s0*s1*yy + c1*zz+deltaz
    
    #New x,y,z axes
    x3= (c0*c1*c2 - s0*s2)*xx + trans1
    y3= (c0*c1*s2 + s0*c2)*xx + trans2
    z3= -c0*s1*xx + trans3
    
    #radius values
    r3 = (x3**2 + y3**2 + z3**2)**0.5
    
    #
    # Density distributions
    #
    
    #Gaussian ring model of disc with peak radius, width
    density_t = np.exp(-0.5*(r3 - radius)**2/rsigma**2) * np.exp(-0.5*((z3/(scaleheight*r3))**2))
    density_t[nc,nc,nc] = 0.0
    
    #Sum disc flux along line of sight
    transformed = np.sum(density_t,axis=2)
    
    #Rescale the image to the same angular scale as the observation
    rescaled = rescale(transformed, rscale / (dstar*pixscale),order=3,mode='constant',cval=0.0,anti_aliasing=True,multichannel=False) #factor of 1.6 is for pixel scale of output images
    #downsample = int((dstar*pixscale)/rscale)
    #rescaled = block_reduce(transformed,(downsample, downsample),np.sum,cval=0.0)
    
    #Issues if the reduced resolution array is even in size    
    if rescaled.shape[0] % 2 == 0:
        rescaled = np.pad(rescaled,(1,1),'constant',constant_values=(0.0,0.0))
        rescaled = shift(rescaled,0.5)
        rescaled = rescaled[1:,1:]
    
    #Now stick the resized model array into a 61x61 grid
    if rescaled.shape[0] < 61:
        ncen = int((rescaled.shape[0]-1)/2)
        padding = int((61-rescaled.shape[0]) / 2)
        rescaled = np.pad(rescaled,(padding,padding),'constant',constant_values=(0.0,0.0))
    else:
        ncen = int((rescaled.shape[0]-1) / 2)
        rescaled = rescaled[ncen-30:ncen+31,ncen-30:ncen+31]
    
    model = fscale * (rescaled/ np.sum(rescaled)) #scale value of rescaled array to total emission from disc
    
    model[30,30] = model[30,30] + fstar #add stellar flux to central pixel of the model image
    #print(np.max(model),np.min(model))
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    #plt.imshow(model)
    
    return model

def lnprior(theta,fdisc,rdisc):
    f,r,d,i,p,dx,dy = theta
    
    if -3 < f < 3 and np.log10(0.325*rdisc) < r < np.log10(1.5*rdisc) and 0.1 < d < 0.9 and 0. < i < 1. and -1. < p < 1. and -6.4 < dx < 6.4 and -6.4 < dy < 6.4:
        return 0.0
    
    return -np.inf

def lnlike(theta,data,y,yerr,kernel,pixscale,fstar,fdisc):
    f,r,d,i,p,dx,dy = theta
    #print(theta)
    #make model
    model = make_disc(data,theta,fstar,pixscale=pixscale)
    #convolve model with PSF
    model = convolve(model,kernel)
    
    #estiamte noise in image
    xi = 1.6*(np.arange((61)) - 30)
    yi = 1.6*(np.arange((61)) - 30)
    xx,yy = np.meshgrid(xi,yi) 
    rr = (xx**2 + yy**2)**0.5
    
    a = np.where((rr > 40) & (rr < 50))
    
    noise = np.std(y[a])
    bkgnd = np.median(y[a])

    #add background to model    
    model = model + bkgnd
    
    #mask source area + a little bit
    zmask = np.zeros((61,61),dtype='float') 
    zmask[np.where((rr < 2*fwhm) | (y > 3*noise))] = 1.
    z2mask = convolve(zmask,kernel)
    mask3 = np.where(z2mask >= 0.10)
    #print((np.sum(((y[mask3] - model[mask3])**2)/(yerr[mask3]**2)) / (len(mask3[0]) - 1)))
    return -0.5 * np.sum(((y[mask3] - model[mask3])/yerr[mask3])**2) 
    
def lnprob(theta, data, y, yerr,kernel,pixscale,fstar,fdisc,rdisc):
    lp=lnprior(theta,fdisc,rdisc)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, y, yerr,kernel,pixscale,fstar,fdisc)

def run_emcee(sampler,pos,ndim,labels,steps=500,prefix=""):
    print("Running MCMC...")
    sampler.run_mcmc(pos,steps,rstate0=np.random.get_state())
    print("Done.")

    plt.clf()
    
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time_v5.png")
    plt.close()
    
    return sampler

def mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    samples[:,0] = 10**samples[:,0] #fdisc in linear
    samples[:,1] = 10**samples[:,1] #rdisc in linear
    samples[:,3] = (180./np.pi)*np.arccos(samples[:,3])
    samples[:,4] = (180./np.pi)*np.arccos(samples[:,4])
    
    #print(samples.shape)
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
    
    ranges = [(credible_interval[0][1]-3.*credible_interval[0][0],credible_interval[0][1]+3.*credible_interval[0][2]),\
              (credible_interval[1][1]-3.*credible_interval[1][0],credible_interval[1][1]+3.*credible_interval[1][2]),\
              (credible_interval[2][1]-3.*credible_interval[2][0],credible_interval[2][1]+3.*credible_interval[2][2]),\
              (credible_interval[3][1]-3.*credible_interval[3][0],credible_interval[3][1]+3.*credible_interval[3][2]),\
              (credible_interval[4][1]-3.*credible_interval[4][0],credible_interval[4][1]+3.*credible_interval[4][2]),\
              (credible_interval[5][1]-3.*credible_interval[5][0],credible_interval[5][1]+3.*credible_interval[5][2]),\
              (credible_interval[6][1]-3.*credible_interval[6][0],credible_interval[6][1]+3.*credible_interval[6][2])]
    fig = corner.corner(samples, color='blue',labels=labels[0:ndim],quantiles=[0.16, 0.5, 0.84],show_titles=True,cmap='blues',range=ranges)
    fig.savefig(prefix+"line-triangle_v5.png")
    plt.close()
    
    print("MCMC results:")
    for i in range(ndim):
        print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))

    #now produce output plots of the images
    theta_bf = [np.log10(credible_interval[0][1]),
                np.log10(credible_interval[1][1]),
                credible_interval[2][1],
                np.cos((np.pi/180.)*credible_interval[3][1]),
                np.cos((np.pi/180.)*credible_interval[4][1]),
                credible_interval[5][1],
                credible_interval[6][1]]
    
    print(theta_bf)
    
    xi = 1.6*(np.arange((61)) - 30)
    yi = 1.6*(np.arange((61)) - 30)
    xx,yy = np.meshgrid(xi,yi) 
    rr = (xx**2 + yy**2)**0.5
    a = np.where((rr > 40) & (rr < 50))
    
    noise = np.std(y[a])
    bkgnd = np.median(y[a])
    
    model_bf = make_disc(data,theta_bf,fstar,pixscale=pixscale)
    convolved = convolve(model_bf,kernel)
    residual = obsvn - (convolved + bkgnd)
    zmask = np.zeros((61,61))
    zmask[np.where((rr < 2*fwhm) | (obsvn > 3*noise))] = 1.
    czmask = convolve(zmask,kernel)
        
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax = axes.ravel()
    ax[0].imshow(obsvn, cmap='rainbow',vmin=np.min(obsvn),vmax=np.max(obsvn),origin='lower')
    ax[0].set_xticks([0,10,20,30,40,50,60])
    ax[0].set_xticklabels(['+48','+32','+16','0','-16','-32','-48'],fontsize=8)
    ax[0].set_yticks([0,10,20,30,40,50,60])
    ax[0].set_yticklabels(['-48','-32','-16','0','+16','+32','+48'],fontsize=8)
    ax[0].text(2,50,r"Observation",fontsize=12,color='black')
    
    ax[1].imshow(model_bf, cmap='rainbow',vmin=0.0,vmax=np.max(model_bf),origin='lower')
    ax[1].set_xticks([0,10,20,30,40,50,60])
    ax[1].set_xticklabels(['+48','+32','+16','0','-16','-32','-48'],fontsize=8)
    ax[1].set_yticks([0,10,20,30,40,50,60])
    ax[1].set_yticklabels(['-48','-32','-16','0','+16','+32','+48'],fontsize=8)
    ax[1].text(2,50,r"Model",fontsize=12,color='black')
    
    ax[2].imshow(convolved, cmap='rainbow',vmin=np.min(obsvn),vmax=np.max(obsvn),origin='lower')
    ax[2].set_xticks([0,10,20,30,40,50,60])
    ax[2].set_xticklabels(['+48','+32','+16','0','-16','-32','-48'],fontsize=8)
    ax[2].set_yticks([0,10,20,30,40,50,60])
    ax[2].set_yticklabels(['-48','-32','-16','0','+16','+32','+48'],fontsize=8)
    ax[2].text(2,50,r"Convolved",fontsize=12,color='black')    
    
    ax[3].imshow(residual, cmap='rainbow',vmin=-3.*noise,vmax=3.*noise,origin='lower')
    ax[3].set_xticks([0,10,20,30,40,50,60])
    ax[3].set_xticklabels(['+48','+32','+16','0','-16','-32','-48'],fontsize=8)
    ax[3].set_yticks([0,10,20,30,40,50,60])
    ax[3].set_yticklabels(['-48','-32','-16','0','+16','+32','+48'],fontsize=8)
    ax[3].text(2,50,r"Residual",fontsize=12,color='black')

    ax[4].imshow(zmask, cmap='rainbow',vmin=0.10,vmax=1.0,origin='lower')
    ax[4].set_xticks([0,10,20,30,40,50,60])
    ax[4].set_xticklabels(['+48','+32','+16','0','-16','-32','-48'],fontsize=8)
    ax[4].set_yticks([0,10,20,30,40,50,60])
    ax[4].set_yticklabels(['-48','-32','-16','0','+16','+32','+48'],fontsize=8)
    ax[4].text(2,50,r"Mask",fontsize=12,color='black')
    
    ax[5].set_visible(False)
    
    fig.text(0.5, 0.04, r'$\Delta$R.A. Offset ($^{\prime\prime}$)', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\Delta$Dec. Offset ($^{\prime\prime}$)', va='center', rotation='vertical')
    fig.savefig(direc+prefix+'residual_disc_v5.png',dpi=200)
    plt.close()

#read in disc details - names, radii (assumed dr = 0.2*r)
jsondirec = '/home/jmarshall/ring_model/json/'
converters = {'Distance': [ascii.convert_numpy(np.float32)],
              'Fstar': [ascii.convert_numpy(np.float32)],
              'Ftotal': [ascii.convert_numpy(np.float32)],
              'radius': [ascii.convert_numpy(np.float32)]}

summary  = ascii.read(jsondirec+'Herschel_Resolved_Sample_Details.csv',converters=converters)

targets       = summary['Target'].data
distance      = summary['Distance'].data
total_flux    = summary['Ftotal'].data
stellar_flux  = summary['Fstar'].data
filename      = summary['Filename'].data
obsids        = summary['ObsIDs'].data
r0            = summary['radius'].data

for i in range(0,1):
    with open(jsondirec+targets[i]+'_disc_model_parameters.json') as json_file:  
        data = json.load(json_file)
    
    print("Analysing target: "+targets[i])
    
    #stellar parameters
    lstar = data["main_results"][0]["lstar"]
    dstar = 1./data["main_results"][0]["plx_arcsec"]
    tstar = data["main_results"][0]["Teff"]
    obs_photosphere_lambda = np.array(data['star_spec']['wavelength'])
    obs_photosphere_flux   = np.array(data['star_spec']['fnujy'])
        
    #fluxes
    phot_band = np.array(data["phot_band"][0])
    obs_lambda = np.array(data["phot_wavelength"][0])
    obs_flux   = np.array(data["phot_fnujy"][0])
    obs_uncs   = np.array(data["phot_e_fnujy"][0])
    obs_ignore = np.array(data["phot_ignore"][0])
    
    #read in fits image
    hdul = fits.open(direc+'fits/'+targets[i]+'/'+filename[i])
    hdr = hdul[0].header #observation information in first element of fits file
    rollangle_obs = hdr['POSANGLE'] #check angle to rotate psf
    #camera = hdr['BLUEBAND'] #check wavelength of observation to use correct psf in convolution
    pixscale = hdr['PIXSIZE'] #pixel scale in arcseconds
    image = hdul[1].data #image is in second element of fits file 
    error = hdul[3].data #error is in fourth element of fits file
    #find the peak of the source in the image - assume it's near the centre of the image
    ind = [int(image.shape[0]/2),int(image.shape[1]/2)] # centre of the image
    crop = image[ind[0]-30:ind[0]+31,ind[1]-30:ind[1]+31]
    guesstimate = fit_2dgaussian(crop) #calculate sub pixel shifts in peak
    cmax = np.unravel_index(np.argmax(crop),crop.shape)
    cx = int(ind[0] - 30 + cmax[0])
    cy = int(ind[1] - 30 + cmax[1])
    
    #centre on somewhere other than peak of stellar emission if it's Fomalhaut.
    if targets[i] == 'HD216956':
        cx = int(284)
        cy = int(254)
    
    obsvn = image[cx-30:cx+31,cy-30:cy+31]
    error = error[cx-30:cx+31,cy-30:cy+31]

    #print(np.unravel_index(np.argmax(obsvn),obsvn.shape),ind,cmax)
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    #
    #ax = axes.ravel()
    #
    #ax[0].imshow(image, cmap='rainbow',vmin=0.1*np.max(image),vmax=0.9*np.max(image))
    #ax[0].set_title("Total Observation")
    #
    #ax[1].imshow(obsvn, cmap='rainbow',vmin=0.1*np.max(obsvn),vmax=0.9*np.max(obsvn))
    #ax[1].set_title("Cropped observatation")
    #
    #fig.savefig(direc+'observed_disc.png',dpi=200)
    #
    #fits.writeto(direc+'obsvn.fits',obsvn,overwrite=True)
    
    band = 'PACS70' in filename[i]
    if band == True:
        camera = 'blue1'
        fwhm = 5.7
    else:
        camera = 'blue2'
        fwhm = 6.8
    
    guesstimate = fit_2dgaussian(obsvn)
    if abs(guesstimate.x_stddev.value) > abs(guesstimate.y_stddev.value):
        semimajor_axis = abs(guesstimate.x_stddev.value)
        semiminor_axis = abs(guesstimate.y_stddev.value)
    else:
        semimajor_axis = abs(guesstimate.y_stddev.value)
        semiminor_axis = abs(guesstimate.x_stddev.value)
    radius = 0.5*dstar*((1.6*2.355*semimajor_axis)**2 - fwhm**2)**0.5
    
    if radius > 500. or np.isnan(radius) == True:
        radius = 0.5*dstar*fwhm
        print(targets[i],"Disc too large or too small")
    posa = abs((180/np.pi)*guesstimate.theta.value)
    incl = (180/np.pi)*np.arccos(semiminor_axis/semimajor_axis)
    
    if targets[i] == 'HD23484':
        radius = 50.0
    if targets[i] == 'HD22049':
        radius = 50.0
    
    if 0. < posa < 180.: 
        print(targets[i]," position angle within bounds, moving on...")
    else:
        posa = 180. - (posa % 180)
        
    #photometry for scaling the disc
    func = interpolate.interp1d(obs_photosphere_lambda,obs_photosphere_flux)
    if camera == 'blue1':
        fstar = float(func(7.0e+01))
        ftotal = obs_flux[np.where((obs_ignore == False) & (phot_band == 'PACS70'))][0]
    else:
        fstar = float(func(1.00e+02))
        ftotal = obs_flux[np.where((obs_ignore == False) & (phot_band == 'PACS100'))][0]
    fdisc = ftotal - fstar # Jy - scaling factor for star - total flux 
    
    #read in PSFs
    psf_direc = direc + 'psfs/' 
    hdul     = fits.open(psf_direc+'HPACS_PSF_Blue.fits')
    hdr = hdul[0].header
    rollangle_blue  = hdr['POSANGLE'] #POSANGLE keyword from FITS Header
    hdr = hdul[1].header
    nx = int(hdr['NAXIS2']/2) # Image size keyworks from FITS header
    ny = int(hdr['NAXIS1']/2) # Image size keyworks from FITS header
    psf_blue = hdul[1].data
    np.nan_to_num(psf_blue,copy=False)
    psf_blue = rotate(psf_blue,rollangle_blue-rollangle_obs,reshape=False)
    
    crop = psf_blue[nx-30:nx+31,ny-30:ny+31]
    guesstimate = fit_2dgaussian(crop) #calculate sub pixel shifts in peak
    cmax = np.unravel_index(np.argmax(crop),crop.shape)
    cx = int(nx - 30 + cmax[0])
    cy = int(ny - 30 + cmax[1])
    psf_blue = psf_blue[cx-30:cx+31,cy-30:cy+31]
    unrolledPSF_blue = psf_blue/np.sum(psf_blue)
    psfblueKernel = CustomKernel(unrolledPSF_blue)
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    #plt.imshow(unrolledPSF_blue)
    #fits.writeto(direc+'psf_blue.fits',unrolledPSF_blue,overwrite=True)
    
    hdul     = fits.open(psf_direc+'HPACS_PSF_Green.fits')
    hdr = hdul[0].header
    rollangle_green  = hdr['POSANGLE'] #POSANGLE keyword from FITS Header
    hdr = hdul[1].header
    nx = int(hdr['NAXIS2']/2) # Image size keyworks from FITS header
    ny = int(hdr['NAXIS1']/2) # Image size keyworks from FITS header
    psf_green = hdul[1].data
    np.nan_to_num(psf_green,copy=False)
    psf_green = rotate(psf_green,rollangle_green-rollangle_obs,reshape=False)
    
    crop = psf_green[nx-30:nx+31,ny-30:ny+31]
    guesstimate = fit_2dgaussian(crop) #calculate sub pixel shifts in peak
    cmax = np.unravel_index(np.argmax(crop),crop.shape)
    cx = int(nx - 30 + cmax[0])
    cy = int(ny - 30 + cmax[1])
    unrolledPSF_green = psf_green[cx-30:cx+31,cy-30:cy+31]
    unrolledPSF_green = unrolledPSF_green/np.sum(unrolledPSF_green)
    psfgreenKernel = CustomKernel(unrolledPSF_green)
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    #plt.imshow(unrolledPSF_green)
    #fits.writeto(direc+'psf_green.fits',unrolledPSF_green,overwrite=True)
        
    #assign PSF kernel for convolution
    kernel = psfblueKernel
    if camera == 'blue2':
        kernel = psfgreenKernel
    
    #estiamte noise in image
    xi = pixscale*(np.arange((61)) - 30)
    yi = pixscale*(np.arange((61)) - 30)
    xx,yy = np.meshgrid(xi,yi) 
    rr = (xx**2 + yy**2)**0.5
    
    noise = np.std(obsvn[np.where((rr > 40.) & (rr < 50.))])
    print(noise)
    mask = np.where((rr < 2*fwhm) | (obsvn > 3.*noise))
    
    #mask source area + a little bit
    zmask = np.zeros((61,61),dtype='float') 
    zmask[mask] = 1.
    
    z2mask = convolve(zmask,kernel)
    mask3 = np.where((z2mask >= 0.10))
    
    #check position angle from 2D Gaussian fit is actually the best one, or if it's rotated by 90 degrees
    pa1 = posa
    print("Postion angle: ",posa," degrees")
    if posa >= 90.:
        pa2 = pa1 - 90.
    else:
        pa2 = pa1 + 90.
    
    thet1 = [np.log10(fdisc),np.log10(radius),0.3,np.cos((np.pi/180.)*incl),np.cos((np.pi/180.)*pa1),0.0,0.0]
    test1 = make_disc(data,thet1,fstar,pixscale=1.6)
    test1 = convolve(test1,kernel)
    resi1 = np.sum(((obsvn[mask3] - test1[mask3])/error[mask3])**2)
    
    thet2 = [np.log10(fdisc),np.log10(radius),0.3,np.cos((np.pi/180.)*incl),np.cos((np.pi/180.)*pa2),0.0,0.0]
    test2 = make_disc(data,thet2,fstar,pixscale=pixscale)
    test2 = convolve(test2,kernel)
    resi2 = np.sum(((obsvn[mask3] - test2[mask3])/error[mask3])**2)
    
    print("Residuals for the two position angles: ",resi1,resi2)
    
    if resi1 <= resi2:
        posa = pa1
    else:
        posa = pa2
    
    #run emcee
    nwalkers = 250
    nsteps = 500
    nburns = 400
    ndim = 7
    y = obsvn
    yerr = error
    
    lograd = np.log10(radius)
    logfdi = np.log10(fdisc)
    coincl = np.cos((np.pi/180.)*incl)
    coposa = np.cos((np.pi/180.)*posa)
    
    pos=[[logfdi + 0.10*np.random.randn(), \
          lograd + 0.10*np.random.randn(), \
          0.2 + 0.10*np.random.uniform(-1,1), \
          coincl + 0.05*np.random.uniform(-1,1), \
          coposa + 0.05*np.random.uniform(-1,1), \
          0.0 + 0.1*np.random.randn(), \
          0.0 + 0.1*np.random.randn()]  for i in range(nwalkers)]
    
    labels=[r"$f_{\rm d}$",r"$R_{\rm d}$",r"$\Delta R_{\rm d}$",r"$\theta$",r"$\phi$",r"$\Delta$x",r"$\Delta$y"]
    with mp.ProcessPool(30) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, moves=emcee.moves.StretchMove(a=2.0),args=(data, y, yerr,kernel,pixscale,fstar,fdisc,radius))
        start = time.time()
        results = run_emcee(sampler,pos,ndim,labels[0:ndim],nsteps,prefix=targets[i]+"_myemcee_v5_")
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    dill.dump(results,open(targets[i]+"_sampler_output_v5.p","wb"))
    mcmc_results(results,ndim,labels=labels,burnin=nburns,prefix=targets[i]+"_myemcee_v5_")