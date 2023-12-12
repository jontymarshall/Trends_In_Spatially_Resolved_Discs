#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:32:34 2019

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from scipy.ndimage.interpolation import rotate 
from scipy.ndimage.interpolation import shift
from scipy import interpolate
#from photutils.centroids import fit_2dgaussian
from skimage.transform import rescale
import json
import emcee
import corner
import dill
import os

#disc model function 
def make_disc(data,theta,fstar):
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
    #print(theta)
    #test disc parameters
    radius = r  # au - radius of peak disc flux in debris belt
    dr     = d  # fractional width dR/R of debris belt
    incl   = i  # degrees
    posang = p  # degrees + 90 because rotation starts at positive x axis
    fscale = f  # mJy - scaling factor for disc - total flux
    
    rwidth  = dr*radius # au - width of debris belt (Gaussian sigma)
    rscale = 5.0 # au per pixel in model image
    if dstar < 10.:
        rscale = 2.0 # au per pixel in model image
    if dstar >= 100.:
        rscale = 10.0 # au per pixel in model image
    scaleheight = 0.1 # open angle for the disc
    
    #calculate model grid for dust density calculation    
    nel = 125
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
    density_t = np.exp(-0.5*((r3 - radius)/rwidth)**2) * np.exp(-((abs(z3)/r3)**2/scaleheight**2)/2.0)
    density_t[nc,nc,nc] = 0.0
    #print(np.max(np.exp(-((abs(z3)/r3)**2/scaleheight**2)/2.0),np.min(np.exp(-((abs(z3)/r3)**2/scaleheight**2)/2.0))))
    #Sum disc flux along line of sight
    transformed = np.sum(density_t,axis=2)
    
    #Rescale the image to the same angular scale as the observation
    rescaled = rescale(transformed, rscale / (dstar*1.6),mode='constant',cval=0.0,multichannel=False) #factor of 1.6 is for pixel scale of output images
    
    #Issues if the reduced resolution array is even in size    
    if rescaled.shape[0] % 2 == 0:
        rescaled = np.pad(rescaled,(1,1),'constant',constant_values=(0.0,0.0))
        rescaled = shift(rescaled,0.5)
        rescaled = rescaled[1:,1:]
    
    #Now stick the resized model array into a 61x61 grid
    if rescaled.shape[0] < 61:
        ncen = int((rescaled.shape[0]-1)/2)
        rescaled[ncen,ncen] = rescaled[ncen,ncen]
        padding = int((61-rescaled.shape[0]) / 2)
        rescaled = np.pad(rescaled,(padding,padding),'constant',constant_values=(0.0,0.0))
    else:
        ncen = int((rescaled.shape[0]-1) / 2)
        rescaled[ncen,ncen] = rescaled[ncen,ncen]
        rescaled = rescaled[ncen-30:ncen+31,ncen-30:ncen+31]
    
    model = (fscale / np.sum(rescaled)) * rescaled #scale value of rescaled array to total emission from disc
    
    model[30,30] = model[30,30] + fstar #add stellar flux to central pixel of the model image
    #print(np.max(model),np.min(model))
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    #plt.imshow(model)
    
    return model

def lnprior(theta,fdisc,rdisc):
    f,r,d,i,p,dx,dy = theta
    
    if 0.80*fdisc < f < 1.20*fdisc and 0.55*fwhm*dstar < r < 2.0*rdisc and 0.1 < d < 0.9 and 0. < i < 90. and 0. < p < 180. and -3.2 < dx < 3.2 and -3.2 < dy < 3.2:
        return 0.0
    
    return -np.inf

def lnlike(theta,data,y,yerr,kernel,fstar,fdisc):
    f,r,d,i,p,dx,dy = theta
    #print(theta)
    #make model
    model = make_disc(data,theta,fstar)
    #convolve model with PSF
    model = convolve(model,kernel)
    #model = model + np.median(y[mask])
    
    # std1 = np.std(image[np.where(image != 0.)])
    # med1 = np.median(image[np.where(image != 0.)])
    # mask1 = np.where((image < 3*std1 + med1))

    # std2 = np.std(image[mask1])
    # med2 = np.median(image[mask1])
    # mask2 = np.where((y > 3*std2))

    #print((np.sum(((y[mask3] - model[mask3])**2)/(yerr[mask3]**2)) / (len(mask3[0]) - 1)))
    return -0.5 * np.sum(((y[mask3] - model[mask3])/yerr[mask3])**2) 

def lnprob(theta, data, y, yerr,kernel,fstar,fdisc,rdisc):
    lp=lnprior(theta,fdisc,rdisc)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, y, yerr,kernel,fstar,fdisc)

def run_emcee(sampler,pos,ndim,labels,steps=500,prefix=""):
    print("Running MCMC...")
    sampler.run_mcmc(pos,steps, rstate0=np.random.get_state())
    print("Done.")

    plt.clf()
    
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time_RFinal.png")
    return sampler

def mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    print(samples.shape)
    
    fig = corner.corner(samples, color='blue',labels=labels[0:ndim],quantiles=[0.16, 0.5, 0.84],show_titles=True,cmap='blues')
    fig.savefig(prefix+"line-triangle_RFinal.png")
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
    
    print("MCMC results:")
    for i in range(ndim):
        print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))

    #now produce output plots of the images
    theta_bf = [credible_interval[0][1],credible_interval[1][1],credible_interval[2][1],credible_interval[3][1],credible_interval[4][1]]
    print(theta_bf)
    model = make_disc(data,theta_bf,fstar,fdisc)
    convolved = convolve(model,kernel)
    residual = obsvn - convolved
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.ravel()
    
    ax[0].imshow(obsvn, cmap='rainbow',vmin=np.min(obsvn),vmax=np.max(obsvn))
    ax[0].set_title("Observation")
    
    ax[1].imshow(model, cmap='rainbow',vmin=np.min(model),vmax=np.max(model))
    ax[1].set_title("Model")
    
    ax[2].imshow(convolved, cmap='rainbow',vmin=np.min(obsvn),vmax=np.max(obsvn))
    ax[2].set_title("Model")    
    
    ax[3].imshow(residual, cmap='rainbow',vmin=-3.*noise,vmax=3.*noise)
    ax[3].set_title("Residual")
    
    fig.savefig(direc+prefix+'residual_disc.png',dpi=200)

#Read in disc data
direc = '/Users/jonty/mydata/ciska/ring_model/'
data_direc = '/Users/jonty/mydata/sasha/resolved_disks_II/targets/' 

#read in disc details - names, radii (assumed dr = 0.2*r)
converters = {'Distance': [ascii.convert_numpy(np.float32)],
              'Fstar': [ascii.convert_numpy(np.float32)],
              'Ftotal': [ascii.convert_numpy(np.float32)],
              'radius': [ascii.convert_numpy(np.float32)]}

summary  = ascii.read(direc+'Herschel_Resolved_Sample_Details.csv',converters=converters)

targets       = summary['Target'].data
distance      = summary['Distance'].data
total_flux    = summary['Ftotal'].data
stellar_flux  = summary['Fstar'].data
filename      = summary['Filename'].data
obsids        = summary['ObsIDs'].data
r0            = summary['radius'].data

#read in results
nwalkers = 250
nsteps = 500
nburns = 400
ndim = 7
labels=[r"$f_{\rm d}$ (mJy)",
        r"$R_{\rm d}$ (au)",
        r"$\Delta R_{\rm d}$",
        r"$\theta$ $^{\circ}$",
        r"$\phi$ $^{\circ}$",
        r"$\Delta$x ($^{\prime\prime}$)",
        r"$\Delta$y ($^{\prime\prime}$)"]

fobs = []
ufobs = []

fdisc  = []
epfdisc = []
enfdisc = []
fstar  = []
radii  = []
epradii = []
enradii = []
uradi  = []
epuradi = []
enuradi = []
incl   = []
epincl = []
enincl = []
posa   = []
epposa = []
enposa = []
lumin  = []
eplstar = []
enlstar = []
distn  = []
epdistn = []
endistn = []
names  = [] 
wavelength = []
fwhm = []
feh = []
tau = []
lam0 = []
beta = []
tstar = []
etstar = []
tdust = []
etdust = []
rbb_disc = []
erbb_disc = []
#two component discs
AB_analogues = []
nwarm = 0

for i in range(len(targets)):
    if os.path.isfile(direc+'v5_results/'+targets[i]+'_sampler_output_v5.p') == 1:
        
        with open(data_direc+targets[i]+'_disc_model_parameters.json') as json_file:  
            data = json.load(json_file)
            
        #stellar data
        lstar = data["main_results"][0]["lstar"]
        eplstar.append(data["main_results"][0]["e_lstar_hi"])
        enlstar.append(data["main_results"][0]["e_lstar_lo"]) 
        dstar = 1./data["main_results"][0]["plx_arcsec"]
        epdistn.append(1./(data["main_results"][0]["plx_arcsec"] - data["main_results"][0]["e_plx_arcsec"]) - dstar)
        endistn.append(dstar - 1./(data["main_results"][0]["plx_arcsec"] + data["main_results"][0]["e_plx_arcsec"]))
        tstar.append(data["main_results"][0]["Teff"])
        etstar.append(data["main_results"][0]["e_Teff"])
        feh.append(data["main_results"][0]["MH"])
        try:
            t1 = data["main_results"][1]["Temp"]
            t2 = data["main_results"][2]["Temp"]
            #print(targets[i],t1,t2)
            if t1 > t2 and t2 < 20.:
                #print("Here1")
                tau.append(data["main_results"][1]["ldisk_lstar"])
                lam0.append(data["main_results"][1]["lam0"])
                beta.append(data["main_results"][1]["beta"])
                tdust.append(data["main_results"][1]["Temp"])
                etdust.append(data["main_results"][1]["e_Temp"])
                rbb_disc.append(data["main_results"][1]["rdisk_bb"])
                erbb_disc.append(data["main_results"][1]["e_rdisk_bb"])
                #nwarm = nwarm+1
                AB_analogues.append(0)
            if t1 < t2 and t1 > 20.:
                #print("Here2")
                tau.append(data["main_results"][1]["ldisk_lstar"])
                lam0.append(data["main_results"][1]["lam0"])
                beta.append(data["main_results"][1]["beta"])
                tdust.append(data["main_results"][1]["Temp"])
                etdust.append(data["main_results"][1]["e_Temp"])
                rbb_disc.append(data["main_results"][1]["rdisk_bb"])
                erbb_disc.append(data["main_results"][1]["e_rdisk_bb"])
                nwarm = nwarm+1
                AB_analogues.append(1)
            if t1 < t2 and t1 < 20.:
                #print("Here3")
                tau.append(data["main_results"][2]["ldisk_lstar"])
                lam0.append(data["main_results"][2]["lam0"])
                beta.append(data["main_results"][2]["beta"])
                tdust.append(data["main_results"][2]["Temp"])
                etdust.append(data["main_results"][2]["e_Temp"])
                rbb_disc.append(data["main_results"][2]["rdisk_bb"])
                erbb_disc.append(data["main_results"][2]["e_rdisk_bb"])
                #nwarm = nwarm+1
                AB_analogues.append(0)
            if t2 < t1 and t2 > 20.:
                #print("Here5")
                tau.append(data["main_results"][2]["ldisk_lstar"])
                lam0.append(data["main_results"][2]["lam0"])
                beta.append(data["main_results"][2]["beta"])
                tdust.append(data["main_results"][2]["Temp"])
                etdust.append(data["main_results"][2]["e_Temp"])
                rbb_disc.append(data["main_results"][2]["rdisk_bb"])
                erbb_disc.append(data["main_results"][2]["e_rdisk_bb"])
                nwarm = nwarm+1
                AB_analogues.append(1)
        except: 
            t1 = data["main_results"][1]["Temp"]
            t2 = -1
            #print("Here5")
            tau.append(data["main_results"][1]["ldisk_lstar"])
            lam0.append(data["main_results"][1]["lam0"])
            beta.append(data["main_results"][1]["beta"])
            tdust.append(data["main_results"][1]["Temp"])
            etdust.append(data["main_results"][1]["e_Temp"])
            rbb_disc.append(data["main_results"][1]["rdisk_bb"])
            erbb_disc.append(data["main_results"][1]["e_rdisk_bb"])
            AB_analogues.append(0)
        #print(i,targets[i])
        del t1,t2
        
        #target photometry
        obs_flux = np.array(data["phot_fnujy"][0])
        obs_uncs = np.array(data["phot_e_fnujy"][0])
        obs_ignr = np.array(data["phot_ignore"][0])
        obs_band = np.array(data["phot_band"][0])
        
        obs_photosphere_lambda = np.array(data['star_spec']['wavelength'])
        obs_photosphere_flux   = np.array(data['star_spec']['fnujy'])
        
        lumin.append(data["main_results"][0]["lstar"])
        distn.append(1./data["main_results"][0]["plx_arcsec"])
        
        #observed fluxes
        try:
            fobs.append(float(obs_flux[np.where((obs_band == 'PACS70') & (obs_ignr != 1))]))
            ufobs.append(float(obs_uncs[np.where((obs_band == 'PACS70') & (obs_ignr != 1))]))
            wavelength.append(' 70 & ')
            fwhm.append(5.8)
        except:
            fobs.append(float(obs_flux[np.where((obs_band == 'PACS100') & (obs_ignr != 1))]))
            ufobs.append(float(obs_uncs[np.where((obs_band == 'PACS100') & (obs_ignr != 1))]))
            wavelength.append(' 100 & ')
            fwhm.append(6.7)
        #stellar photospheric flux
        fstar.append(stellar_flux[i])
        
        samples = np.zeros([nwalkers,nsteps,ndim])
        
        try:
            with open(direc+'v5_results/'+targets[i]+'_sampler_output_v5.p', "rb") as input_file:
            #    print(input_file)
                values = dill.load(input_file)
        except: 
            with open(direc+'v4_results/'+targets[i]+'_sampler_output_v4.p', "rb") as input_file:
            #    print(input_file)
                values = dill.load(input_file)

        #print(values.sampler)
        samples = values.chain
        #good = np.where(samples != 0.0)
        #samples = samples[good]
        samples = samples.reshape((-1, ndim))
        
        #convert values into actual (not log space or cos() or whatever)
        samples[:,0] = 10**samples[:,0] #flux
        samples[:,1] = 10**samples[:,1] #radius
        bad = np.where(samples[:,3] > 1.0) 
        samples[bad,3] = samples[bad,3] - 1.0
        samples[:,3] = (180./np.pi)*np.arccos(samples[:,3]) #inclination
        bad = np.where(samples[:,4] > 1.0) 
        samples[bad,4] = samples[bad,4] - 1.0
        bad = np.where(samples[:,4] < -1.0)
        samples[bad,4] = samples[bad,4] + 1.0
        samples[:,4] = (180./np.pi)*np.arccos(samples[:,4]) #position angle
        #print(samples.shape,samples.size)
        #fig = corner.corner(samples, labels=labels[0:ndim])
        #fig.savefig(direc+targets[i]+"_combined_line-triangle_v2.png")
        credible_interval=[]
        percentiles = [16,50,84]
        for j in range(ndim):
            credible_interval.append(np.percentile(samples[nburns-1:,j], percentiles))
            credible_interval[j][2] -= credible_interval[j][1]
            credible_interval[j][0] = credible_interval[j][1] - credible_interval[j][0]
            #print(j, credible_interval[j][1],credible_interval[j][0],credible_interval[j][2])
        #print("MCMC results for "+targets[i]+":")
        #for j in range(ndim):
        #    print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[j],credible_interval[j]))
        fdisc.append(credible_interval[0][1])
        epfdisc.append(credible_interval[0][2])
        enfdisc.append(credible_interval[0][0])
        radii.append(credible_interval[1][1])
        epradii.append(credible_interval[1][2])
        enradii.append(credible_interval[1][0])
        uradi.append(credible_interval[2][1]*credible_interval[1][1])
        epuradi.append(credible_interval[2][2]*credible_interval[1][1])
        enuradi.append(credible_interval[2][0]*credible_interval[1][1])
        incl.append(credible_interval[3][1])
        epincl.append(credible_interval[3][2])
        enincl.append(credible_interval[3][0])        
        posa.append(credible_interval[4][1])
        epposa.append(credible_interval[4][2])
        enposa.append(credible_interval[4][0])  
        names.append(targets[i])
        
        del data
        
    else:
        pass

#plot up results
names = np.array(names)
distn = np.array(distn)
lumin = np.array(lumin)
feh   = np.array(feh)
fdisc = np.array(fdisc)
radii = np.array(radii) 
epradii = np.array(epradii) 
enradii = np.array(enradii) 
uradi = np.array(uradi)
epuradi = np.array(epuradi) 
enuradi = np.array(enuradi) 
incl  = np.array(incl)
posa  = np.array(posa)
tstar = np.array(tstar)
etstar = np.array(etstar)
tdust = np.array(tdust)
etdust = np.array(etdust)
tau = np.array(tau)
lam0 = np.array(lam0)
beta = np.array(beta)
rbb = np.array(rbb_disc)
erbb = np.array(erbb_disc)

fwhm =  np.array(fwhm)


#Number of asteroid belt analogues in the data set
print("There are ",nwarm," systems with warm components in the sample, based on SED modelling.")
AB_analogues = np.array(AB_analogues)
warm = np.where(AB_analogues != 0)
cold = np.where(AB_analogues == 0)
print(targets[warm])
print("")
#plot up comparison of discs with warm belts and single belt disc.
fig=plt.figure()
ax=fig.add_subplot(111)
#im = ax.scatter(tdust[cold],radii[cold],marker='o',c=tstar[cold],s=uradi[cold],cmap='coolwarm_r')
tstar_warm = np.clip(tstar[warm],3000.,9000.)
im = ax.scatter(tdust[warm],radii[warm],marker='o',c=tstar_warm,s=100.*(uradi[warm]/radii[warm]),cmap='coolwarm_r')
ax.scatter([90,95,100,105],[275,275,275,275],marker='o',color='black',s=[10,25,50,100])
ax.set_xlim([20.0,120.0])
ax.set_ylim([0.0,300.0])
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Disc temperature (K)')
ax.set_ylabel(r'Disc radius (au)')
fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)',ticks=[3000, 4000, 5000, 6000, 7000, 8000, 9000])
Normalize(vmin=3000,vmax=10000)
fig.savefig(direc+"v5_results/"+"R-L_two_belts_bias_warm_publication_RFinal.png",dpi=200)

fig=plt.figure()
ax=fig.add_subplot(111)
tstar_cold = np.clip(tstar[cold],3000.,9000.)
im = ax.scatter(tdust[cold],radii[cold],marker='o',c=tstar_cold,s=100.*(uradi[cold]/radii[cold]),cmap='coolwarm_r')
ax.scatter([90,95,100,105],[275,275,275,275],marker='o',color='black',s=[10,25,50,100])
ax.set_xlim([20.0,120.0])
ax.set_ylim([0.0,300.0])
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Disc temperature (K)')
ax.set_ylabel(r'Disc radius (au)')
fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)',ticks=[3000, 4000, 5000, 6000, 7000, 8000, 9000])
fig.savefig(direc+"v5_results/"+"R-L_two_belts_bias_cold_publication_RFinal.png",dpi=200)

good = np.where((2.*(radii/distn) > 0.5*fwhm))

tbb_rad = 278*radii**(-0.5)*lumin**0.25
tratio = tdust/tbb_rad

fig=plt.figure()
ax=fig.add_subplot(111)

im = ax.scatter(lumin[good],tratio[good],marker='o',c=tstar[good],cmap='coolwarm_r')
ax.set_xlim([0.01,100.0])
ax.set_ylim([0.0,3.0])
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_ylabel(r'$T_{\rm dust}/T_{\rm bb}$')
ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)',ticks=[3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
Normalize(vmin=3000,vmax=10000)
fig.savefig(direc+"v5_results/"+"R-L_disc_temperature_ratio_publication_RFinal.png",dpi=200)


fig=plt.figure()
ax=fig.add_subplot(111)
n_c, bins_c, patches_c = ax.hist(uradi[warm]/radii[warm], facecolor='red', alpha=1,bins=np.arange(0.1, 1.1, 0.05),zorder=2)
n_w, bins_w, patches_w = ax.hist(uradi/radii, facecolor='orange', alpha=0.75,bins=np.arange(0.1, 1.1, 0.05),zorder=1)
ax.set_xlim([0.0,1.0])
ax.set_ylim([0.0,40.0])
ax.set_xlabel(r'Disc fractional width ($W_{\rm disc} = \Delta R_{\rm disc}/R_{\rm disc}$)')
ax.set_ylabel(r'Number')
fig.savefig(direc+"v5_results/"+"R-L_two_belts_bias_width_hist_publication_RFinal.png",dpi=200)

fig=plt.figure()
ax=fig.add_subplot(111)
n_c, bins_c, patches_c = ax.hist(radii[warm], facecolor='red', alpha=1,bins=np.arange(0.0, 300., 25.),zorder=2)
n_w, bins_w, patches_w = ax.hist(radii, facecolor='orange', alpha=0.75,bins=np.arange(0.0, 300., 25.),zorder=1)
ax.set_xlim([0.0,250.0])
ax.set_ylim([0.0,30.0])
ax.set_xlabel(r'Disc radius (au)')
ax.set_ylabel(r'Number')
fig.savefig(direc+"v5_results/"+"R-L_two_belts_bias_radii_hist_publication_RFinal.png",dpi=200)


fig=plt.figure()
ax=fig.add_subplot(111)
n, bins, patches = ax.hist(uradi/radii, facecolor='red', alpha=0.75,bins=np.arange(0.1, 1.1, 0.05))
ax.set_xlim([0.0,1.0])
ax.set_ylim([0.0,20.0])
ax.set_yticks([5,10,15,20])
ax.plot([0.1,0.1],[0.0,20.0],color='black',linestyle='--')
ax.plot([0.3,0.3],[0.0,20.0],color='black',linestyle='--')
ax.set_xlabel(r'Disc fractional width ($W_{\rm disc} = \Delta R_{\rm disc}/R_{\rm disc}$)')
ax.set_ylabel(r'Number')
fig.savefig(direc+"v5_results/"+"R-L_far-infrared_widths_publication_RFinal.png",dpi=200)

#
#
print(len(good[0]))

lumin_string = []
distn_string = []
fobs_string  = []
fstar_string = []
fdisc_string = []
radii_string = []
uradi_string = []
posa_string  = []
incl_string  = []

for i in range(len(names)):
    stra = "{0:#.3f}".format(distn[i])
    strb = "{0:#.3f}".format(epdistn[i])
    strc = "{0:#.3f}".format(endistn[i])
    distn_string.append(' & $'+stra+'^{+'+strb+'}_{-'+strc+'}$ & ')
    stra = "{0:#.3f}".format(lumin[i])
    strb = "{0:#.3f}".format(eplstar[i])
    strc = "{0:#.3f}".format(enlstar[i])
    lumin_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ & ')
    stra = "{0:#.0f}".format(fobs[i]*1000)
    strb = "{0:#.0f}".format(ufobs[i]*1000)
    fobs_string.append('$'+stra+'~\pm~'+strb+'$ & ')
    fstar_string.append('$'+"{0:#.0f}".format(fstar[i])+'$ & ')
    stra = "{0:#.0f}".format(fdisc[i]*1000)
    strb = "{0:#.0f}".format(epfdisc[i]*1000)
    strc = "{0:#.0f}".format(enfdisc[i]*1000)    
    fdisc_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.1f}".format(radii[i])
    strb = "{0:#.1f}".format(epradii[i])
    strc = "{0:#.1f}".format(enradii[i])    
    radii_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.1f}".format(uradi[i])
    strb = "{0:#.1f}".format(epuradi[i])
    strc = "{0:#.1f}".format(enuradi[i])    
    uradi_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    if posa[i] < 90. :
        stra = "{0:#.1f}".format(posa[i]+90.)
    if posa[i] >= 90. : 
        stra = "{0:#.1f}".format(180. - posa[i])
    strb = "{0:#.1f}".format(epposa[i])
    strc = "{0:#.1f}".format(enposa[i])    
    posa_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ &')
    stra = "{0:#.1f}".format(incl[i])
    strb = "{0:#.1f}".format(epincl[i])
    strc = "{0:#.1f}".format(enincl[i])    
    incl_string.append('$'+stra+'^{+'+strb+'}_{-'+strc+'}$ \\\\')
    

lumin_string = np.array(lumin_string)
distn_string = np.array(distn_string)
fobs_string  = np.array(fobs_string)
fstar_string = np.array(fstar_string)
fdisc_string = np.array(fdisc_string)
radii_string = np.array(radii_string)
uradi_string = np.array(uradi_string)
posa_string  = np.array(posa_string)
incl_string  = np.array(incl_string)
wavelength   = np.array(wavelength)

goodsort = np.argsort(lumin)

sortnames = names[goodsort]
distn_string = distn_string[goodsort]
lumin_string = lumin_string[goodsort]
wavelength   = wavelength[goodsort]
fobs_string  = fobs_string[goodsort]
fstar_string = fstar_string[goodsort]
fdisc_string = fdisc_string[goodsort]
radii_string = radii_string[goodsort]
uradi_string = uradi_string[goodsort]
posa_string  = posa_string[goodsort]
incl_string  = incl_string[goodsort]

valid = np.where((2.*(radii[goodsort]/distn[goodsort]) > 0.5*fwhm[goodsort]))

ascii.write([sortnames[valid],distn_string[valid],lumin_string[valid],\
            wavelength[valid],fobs_string[valid],fstar_string[valid],fdisc_string[valid],\
            radii_string[valid],uradi_string[valid],\
            posa_string[valid],incl_string[valid]],\
            direc+'v5_results/'+'emcee_resolved_disc_output_v5_latex_sorted.dat',\
            names=['Name', 'Distance', 'Lstar','Wavelength','Fobs','Fstar','Fdisc','Radius','Width','PosAng','Incl'],overwrite=True )

targets = names[good]
#lstar = lumin[good]
radii = radii[good]
epradii = epradii[good]
enradii = enradii[good]
uradi = uradi[good]
epuradi = epuradi[good]
enuradi = enuradi[good]

tgood = tstar[good]
etgood = etstar[good]
print("---")
print("Target Name, Temperature, Uncertainty in Temperature")
for i in range(0,len(targets)):
    print(targets[i],tgood[i],etgood[i])
print("---")
##Gamma calculation
#gamma = radii/rbb[good]
#egamma = gamma*((erbb[good]/rbb[good])**2 + (epradii/radii)**2)**0.5
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#im = ax.errorbar(lstar,gamma,xerr=None,yerr=egamma,marker='o',mec='blue',mfc='lightblue',linestyle='',elinewidth=0.5,capsize=2,capthick=0.5,color='black',zorder=1)
#ax.set_xlim([0.1,100.0])
#ax.set_ylim([0.5,10.0])
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel(r'Stellar luminosity ($L_{\odot}$)')
#ax.set_ylabel(r'$\Gamma$ ($R_{\rm disc}/R_{\rm bb}$)')
#fig.savefig("R-L_gamma_values_publication.png",dpi=200)
#
#
#Sanity check radii and widths vs. PSF FWHM
lstar = lumin[good]
san_radii = 2.0*radii/distn[good] / 0.5*fwhm[good]
san_epradi = epradii/distn[good] / 0.5*fwhm[good]
san_enradi = enradii/distn[good] / 0.5*fwhm[good]
san_width =  uradi/distn[good] / 0.5*fwhm[good]
san_epwidt = epuradi/distn[good] / 0.5*fwhm[good]
san_enwidt = enuradi/distn[good] / 0.5*fwhm[good]

fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(lstar,san_radii,yerr=[san_epradi,san_enradi],linestyle='',marker='o',mfc="xkcd:salmon",mec='xkcd:red',ecolor='black')
ax.set_xlim([0.01,100.0])
ax.set_ylim([0.0,5.0])
ax.plot([0.01,100.0],[1.0,1.0],linestyle='-',color='black')
ax.plot([0.01,100.0],[1.5,1.5],linestyle='--',color='black')
ax.plot([0.01,100.0],[0.5,0.5],linestyle='--',color='black')
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
ax.set_ylabel(r'(2$\times R_{\rm disc}/d_{\star}$) / PSF HWHM')
fig.savefig(direc+"v5_results/"+"R-L_sanity_radii_publication_RFinal.png",dpi=200)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(lstar,san_width,yerr=[san_epwidt,san_enwidt],linestyle='',marker='s',mfc="xkcd:salmon",mec='xkcd:red',ecolor='black')
ax.set_xlim([0.01,100.0])
ax.plot([0.01,100.0],[1.0,1.0],linestyle='-',color='black')
ax.plot([0.01,100.0],[1.5,1.5],linestyle='--',color='black')
ax.plot([0.01,100.0],[0.5,0.5],linestyle='--',color='black')
ax.set_ylim([0.0,5.0])
ax.set_xscale('log')
ax.set_yscale('linear')
ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
ax.set_ylabel(r'(2$\times \Delta R_{\rm disc}/d_{\star}$) / PSF HWHM')
fig.savefig(direc+"v5_results/"+"R-L_sanity_widths_publication_RFinal.png",dpi=200)
#
#
##Sanity check on biases
#xd = np.arange(0.1,100.0,0.1) #distance in parsecs
#xsi = 5.8*xd #angular size that can be resolved
#xsi50  =  50./xd #angular size of  50 au disc 
#xsi100 = 100./xd #angular size of 100 au disc
#xsi150 = 150./xd #angular size of 150 au disc
#xsitre1 = 87.*1 / xd
#xsitre5 = 87.*(5.**0.19) / xd
#xsitre10 = 87* (10.**0.19) / xd
#xsitre30 = 87* (30.**0.19) / xd
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#im = ax.scatter(distn[good],radii/distn[good],marker='o',c=tstar[good],s=uradi/2.,cmap='coolwarm_r')
##ax.errorbar(distn[good],radii/distn[good],yerr=None,linestyle='',marker='o',mfc="xkcd:grey",mec='xkcd:grey')
#ax.plot(xd,xsi50,linestyle='solid',color='black')
#ax.plot(xd,xsi100,linestyle='dashed',color='black')
#ax.plot(xd,xsi150,linestyle='dotted',color='black')
#ax.plot([0,100],[2.85,2.85],linestyle='solid',color='gray')
#ax.plot([0,100],[3.35,3.35],linestyle='dashed',color='gray')
#ax.scatter([80,85,90,95],[9,9,9,9],marker='o',color='black',s=[10,25,50,100])
#ax.set_xlim([0.0,100.0])
#ax.set_ylim([0.0,10.0])
#ax.set_xscale('linear')
#ax.set_yscale('linear')
#ax.set_xlabel(r'Distance (pc)')
#ax.set_ylabel(r'Disc angular size ($^{\prime\prime}$)')
#fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)')
#fig.savefig(direc+"v5_results/"+"R-L_resolution_bias_angular_publication.png",dpi=200)
#
#
valr = np.linspace(0.0,500.0,1000)
minr = valr*1.35

fig=plt.figure()
ax=fig.add_subplot(111)
im = ax.scatter(distn[good],radii,marker='o',c=tstar[good],s=uradi/2.,cmap='coolwarm_r')
#ax.errorbar(distn[good],radii/distn[good],yerr=None,linestyle='',marker='o',mfc="xkcd:grey",mec='xkcd:grey')
ax.scatter([80,85,90,95],[25,25,25,25],marker='o',color='black',s=[10,25,50,100])
ax.plot(valr,minr,linestyle='-',color='black')
ax.set_xlim([0.0,100.0])
ax.set_ylim([0.0,300.0])
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Distance (pc)')
ax.set_ylabel(r'Disc radius (au)')
fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)')
fig.savefig(direc+"v5_results/"+"R-L_resolution_bias_linear_publication_RFinal.png",dpi=200)
#
#check trend with temperature, overplot Ballering et al. 2013 results for reference
converters = {'T*': [ascii.convert_numpy(np.float32)],
              'Tcold': [ascii.convert_numpy(np.float32)],
              'Twarm': [ascii.convert_numpy(np.float32)]}
data = ascii.read(direc+'ballering_2013.txt',delimiter=';',converters=converters,guess=False,fast_reader=False)

ts_ballering = data['T*'].data
limit = data['l_Tcold'].data
td_ballering = data['Tcold'].data

gd_a = np.where((limit != '<'))

ts_ballering = ts_ballering[gd_a]
td_ballering = td_ballering[gd_a]

gd_b = np.where(td_ballering > 0.)

ts_ballering = ts_ballering[gd_b]
td_ballering = td_ballering[gd_b]

markersize = 4.*np.log10(1e6*tau[good])
print(np.min(markersize),np.max(markersize))
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ts_ballering,td_ballering,marker='o',mec='lightgrey',mfc='lightgrey',linestyle='',markersize=3,zorder=1)
im = ax.scatter(tstar[good],tdust[good],marker='o',c=tstar[good],cmap='coolwarm_r',zorder=3)
ax.errorbar(tstar[good],tdust[good],xerr=None,yerr=etdust[good],marker='',linestyle='',elinewidth=1,capsize=2,capthick=0.5,color='black',zorder=2)
ax.set_xlim([3000.0,11000.0])
ax.set_ylim([0.0,120.0])
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'Stellar temperature (K)')
ax.set_ylabel(r'Disc temperature (K)')
fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)')
fig.savefig("R-L_temperature_trend_publication_RFinal.png",dpi=200)

#do MCMC to fit T_dust vs T_star trend
def lnprior(theta):
    m,b=theta
    if -1 < m < 1 and 0 < b < 1e3:
        return 0.0
    return -np.inf

def lnlike(theta,x,y,yerr):
    m,b=theta
    model= m*x + b
    return -0.5 * np.sum((((y - model)**2)/(yerr**2)))

def lnprob(theta,x,y,yerr):
    lp=lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def run_emcee(sampler,pos,ndim,labels,steps=500,prefix=""):
    print("Running MCMC...")
    sampler.run_mcmc(pos,steps, rstate0=np.random.get_state())
    print("Done.")

    plt.clf()
    
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time_RFinal.png")
    return sampler

#exit()
def mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    print(samples.shape)
    
    fig = corner.corner(samples, color='red',labels=labels[0:ndim],quantiles=[0.16, 0.5, 0.84],show_titles=True,cmap='reds')
    fig.savefig(prefix+"line-triangle_RFinal.png")
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
        #m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                 zip(*np.percentile(samples, percentiles,
        #                                    axis=0)
        #                     )
        #                 )
    #print(quantiles)
    #exit()
    print("MCMC results:")
    for i in range(ndim):
        print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))

    #now produce output plots of the distribution of lines

    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    xplot=np.arange(-2.0,2.0,0.01)
    try:
        for m,b in samples[np.random.randint(len(samples),size=1000),0:2]:
            ax.plot(xplot,m*xplot + b,color="k",alpha=0.01)
    except:
        for m,b in samples[np.random.randint(len(samples),size=1000)]:
            ax.plot(xplot,m*xplot + b,color="k",alpha=0.01)
    ax.errorbar(x,y,yerr=yerr,fmt="ob")
    ax.set_xlim([-1.0,1.5])
    ax.set_ylim([1.0,2.5])
    #ax.set_xscale('log')
    ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
    ax.set_ylabel(r'Radius (au)')
    fig.savefig(prefix+"line-mcmc_RFinal.png")

#MCMC for straight line log(L)-log(R) fit.
A = np.vstack((np.ones_like(tstar[good]), tstar[good])).T
C = np.diag(etdust[good] * etdust[good])
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, tdust[good])))
sigm = np.sqrt(cov[1, 1])
sigb = np.sqrt(cov[0, 0])

nwalkers = 200
nsteps = 500
ndim = 2
x = tstar[good]
y = tdust[good]
yerr = etdust[good]
pos=[[m_ls+ sigm*np.random.randn(),b_ls+sigb*np.random.randn()]  for i in range(nwalkers)]
labels=["$m$","$b$"]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
results = run_emcee(sampler,pos,ndim,labels[0:ndim],nsteps,prefix=direc+"temperature_trend_sample_linefit_myemcee_")
mcmc_results(results,ndim,labels=labels,burnin=100,prefix=direc+"temperature_trend_sample_linefit_myemcee_")

#same but for unresolved debris disc sample

##load unresolved disc host star distances, temperatures, and luminosities
#converters = {'Distance': [ascii.convert_numpy(np.float32)],
#              'Luminosity': [ascii.convert_numpy(np.float32)],
#              'Temperature': [ascii.convert_numpy(np.float32)]}
#
#unresolved  = ascii.read(direc+'Herschel_Unresolved_Sample_Details.csv',converters=converters)
#
#unr_targets = np.array(unresolved['Target'].data)
#unr_lstar   = np.array(unresolved['Luminosity'].data)
#unr_distn   = np.array(unresolved['Distance'].data)
#unr_tstar   = np.array(unresolved['Temperature'].data)
#unr_radii   = 87.*unr_lstar**0.19
#
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.scatter(distn[good],radii/distn[good],marker='o',c='gray',s=50.)
#im = ax.scatter(unr_distn,unr_radii/unr_distn,marker='o',c=unr_tstar,s=50.,cmap='coolwarm_r')
##ax.errorbar(distn[good],radii/distn[good],yerr=None,linestyle='',marker='o',mfc="xkcd:grey",mec='xkcd:grey')
#ax.plot(xd,xsi50,linestyle='solid',color='black')
#ax.plot(xd,xsi100,linestyle='dashed',color='black')
#ax.plot(xd,xsi150,linestyle='dotted',color='black')
#ax.plot([0,100],[2.85,2.85],linestyle='solid',color='gray')
#ax.plot([0,100],[3.35,3.35],linestyle='dashed',color='gray')
#ax.set_xlim([0.0,100.0])
#ax.set_ylim([0.0,10.0])
#ax.set_xscale('linear')
#ax.set_yscale('linear')
#ax.set_xlabel(r'Distance (pc)')
#ax.set_ylabel(r'Disc angular size ($^{\prime\prime}$)')
#fig.colorbar(mappable=im,cmap='coolwarm', ax=ax,label=r'$T_{\star}$ (K)')
#fig.savefig("R-L_resolution_bias2_publication.png",dpi=200)



#Read in the second text file with mm-resolved discs
data  = ascii.read(direc+'mm_resolved_discs_sample.csv')
mm_lst = data["Lstar"].data
mm_rad = data["Radius"].data
mm_wid = 0.5*data["Width"].data

#Plot up data points
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
ax.set_ylabel(r'Radius (au)')
ax.set_xlim([0.01,20.0])
ax.set_ylim([0.0,300.0])
ax.set_xscale('log')
ax.errorbar(lstar,radii,xerr=None,yerr=uradi,marker='o',linestyle='',color='black')
ax.errorbar(mm_lst,mm_rad,xerr=None,yerr=mm_wid,marker='o',linestyle='',color='gray')
fig.savefig(direc+"v5_results/"+'sample_linefit_myfigure1_RFinal.png',dpi=200)


##Fit straight line to slope of far infrared sample
def syn_mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    #print(samples.shape)
    
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
        #m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                 zip(*np.percentile(samples, percentiles,
        #                                    axis=0)
        #                     )
        #                 )
    #print(quantiles)
    #exit()
    #print("MCMC results:")
    #for i in range(ndim):
    #    print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))
    m = credible_interval[0][1]
    b = credible_interval[1][1]
    return m,b

#MCMC for straight line log(L)-log(R) fit.
# $m$  = 0.13490353391732735 + 0.06596712956068296 - 0.062177434174139595
# $b$  = 1.9453643577541506 + 0.060570477495325514 - 0.067659541669437
A = np.vstack((np.ones_like(radii), radii)).T
C = np.diag(uradi * uradi)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, lstar)))
sigm = np.sqrt(cov[1, 1])
sigb = np.sqrt(cov[0, 0])
nwalkers = 200
nsteps = 500
ndim = 2
x = np.log10(lstar)
y = np.log10(radii)
yerr = uradi/radii
pos=[[m_ls+ sigm*np.random.randn(),b_ls+sigb*np.random.randn()]  for i in range(nwalkers)]
labels=["$m$","$b$"]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
results = run_emcee(sampler,pos,ndim,labels[0:ndim],nsteps,prefix=direc+"v5_results/"+"sample_linefit_myemcee_")
mcmc_results(results,ndim,labels=labels,burnin=100,prefix=direc+"v5_results/"+"sample_linefit_myemcee_")

##Test range of uncertainties with mcmc of synthetic observations consistent with
##uncertainties measured from 
nsets = 1000
m_values = []#np.zeros(nsets)
b_values = []#np.zeros(nsets)


#synthetic data
syn_radii = np.zeros(len(radii))
syn_uradi = np.zeros(len(uradi))

for i in range(0,nsets):
    
    #generate data set
    #scattering in measured values
    for j in range(0,len(radii)):
        syn_radii[j] = radii[j] + np.random.uniform(-1*enradii[j],epradii[j])
        syn_uradi[j] = uradi[j] + np.random.uniform(-1*enuradi[j],epuradi[j])
        #print(c,rsca,usca)
    #print(syn_radii,syn_uradi)
    
    #straight line least squares fit
    A = np.vstack((np.ones_like(lstar), lstar)).T
    C = np.diag(syn_uradi * syn_uradi)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, syn_radii)))
    
    #emcee the synthetic data set
    nwalkers = 200
    nsteps = 500
    ndim = 2
    x = np.log10(lstar)
    y = np.log10(syn_radii)
    yerr = syn_uradi/syn_radii
    #print(x,y,yerr)
    pos=[[m_ls+ sigm*np.random.randn(),b_ls+sigb*np.random.randn()]  for i in range(nwalkers)]
    labels=["$m$","$b$"]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos,nsteps, rstate0=np.random.get_state())
    m,b = syn_mcmc_results(sampler,ndim,labels=labels,burnin=100)
    
    m_values.append(m)
    b_values.append(b) 
    #print(m,b)
    
m_values = np.array(m_values)
b_values = np.array(b_values)

print(np.median(m_values),np.std(m_values))
print(np.median(b_values),np.std(b_values))

# ##Publication plots
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(lstar,radii,yerr=uradi,linestyle='',marker='o',mfc="xkcd:salmon",mec='xkcd:red',ecolor='black')
ax.set_xlim([0.01,100.0])
ax.set_ylim([10.0,300.0])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
ax.set_ylabel(r'Disc radius (au)')
xplot=np.arange(0.01,100.0,0.01)
for m,b in samples[np.random.randint(len(samples),size=1000),0:2]:
    ax.plot(xplot,xplot**m * 10**b,color="gray",alpha=0.01)
fig.savefig(direc+"v5_results/"+"R-L_far-infrared_whole_sample_publication_RFinal.png",dpi=200)

##check trend with d < 50 pc subsample
dstar_good = distn[good]
tstar_good = tstar[good]
nearby = np.where(dstar_good <= 40.0)

#raw numbers
nwalkers = 200
nsteps = 200
ndim = 2
x = np.log10(lstar[nearby])
y = np.log10(radii[nearby])
yerr = uradi[nearby]/radii[nearby]
pos=[[m_ls+ sigm*np.random.randn(),b_ls+sigb*np.random.randn()]  for i in range(nwalkers)]
labels=["$m$","$b$"]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
results = run_emcee(sampler,pos,ndim,labels[0:ndim],nsteps,prefix=direc+"nearby_sample_linefit_myemcee_")
mcmc_results(results,ndim,labels=labels,burnin=100,prefix=direc+"nearby_sample_linefit_myemcee_")

#Test range of uncertainties with mcmc of synthetic observations consistent with
#uncertainties measured from 
nsets = 100
m_values = []#np.zeros(nsets)
b_values = []#np.zeros(nsets)

#data in question
lsnear = lstar[nearby]
near_r = radii[nearby]
near_u = uradi[nearby]
ennearr = enradii[nearby]
epnearr = epradii[nearby]
ennearu = enuradi[nearby]
epnearu = epuradi[nearby]

def syn_mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    #print(samples.shape)
    
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
        #m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                 zip(*np.percentile(samples, percentiles,
        #                                    axis=0)
        #                     )
        #                 )
    #print(quantiles)
    #exit()
    #print("MCMC results:")
    #for i in range(ndim):
    #    print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))
    m = credible_interval[0][1]
    b = credible_interval[1][1]
    return m,b

#synthetic data
syn_radii = np.zeros(len(near_r))
syn_uradi = np.zeros(len(near_u))

for i in range(0,nsets):
    
    #generate data set
    #scattering in measured values
    for j in range(0,len(near_r)):
        syn_radii[j] = near_r[j] + np.random.uniform(-1*ennearr[j],epnearr[j])
        syn_uradi[j] = near_u[j] + np.random.uniform(-1*ennearu[j],epnearu[j])
        #print(c,rsca,usca)
    #print(syn_radii,syn_uradi)
    
    #straight line least squares fit
    A = np.vstack((np.ones_like(lsnear), lsnear)).T
    C = np.diag(syn_uradi * syn_uradi)
    cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
    b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, syn_radii)))
    sigm = np.sqrt(cov[1, 1])
    sigb = np.sqrt(cov[0, 0])
    
    #emcee the synthetic data set
    nwalkers = 200
    nsteps = 200
    ndim = 2
    x = np.log10(lsnear)
    y = np.log10(syn_radii)
    yerr = syn_uradi/syn_radii
    #print(x,y,yerr)
    pos=[[m_ls+ sigm*np.random.randn(),b_ls+sigb*np.random.randn()]  for i in range(nwalkers)]
    labels=["$m$","$b$"]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos,nsteps, rstate0=np.random.get_state())
    m,b = syn_mcmc_results(sampler,ndim,labels=labels,burnin=100)
    
    m_values.append(m)
    b_values.append(b) 
    #print(m,b)
    
m_values = np.array(m_values)
b_values = np.array(b_values)

print(np.median(m_values),np.std(m_values))
print(np.median(b_values),np.std(b_values))

samples = sampler.chain[:, 100:, :].reshape((-1, ndim))


fig=plt.figure()
ax=fig.add_subplot(111)
ax.errorbar(lstar[nearby],radii[nearby],yerr=uradi[nearby],linestyle='',marker='o',mfc="xkcd:salmon",mec='xkcd:red',ecolor='black')
ax.set_xlim([0.01,100.0])
ax.set_ylim([10.0,300.0])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Stellar Luminosity (L$_{\odot}$)')
ax.set_ylabel(r'Disc radius (au)')
xplot=np.arange(0.01,100.0,0.01)
for m,b in samples[np.random.randint(len(samples),size=1000),0:2]:
    ax.plot(xplot,xplot**m * 10**b,color="gray",alpha=0.01)
fig.savefig(direc+"v5_results/"+"R-L_far-infrared_distance_limited_bootstrap_publication_RFinal.png",dpi=200)


