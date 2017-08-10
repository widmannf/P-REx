###########
## This file is part of the Python Module P-Rex, a module for test for a 
## piston reconstruction experiement for optical interferometer (see Pott et al 2016)
## This files contains several functions to apply the algorithms on data from a 
## Yao Simulation (by F. Rigaout, frigaut.github.io/yao/index.html)
##
## Copyright (c) 2017, Felix Widmann
##
## This program is free software; you can redistribute it and/or  modify it
## under the terms of the GNU General Public License  as  published  by the
## Free Software Foundation; either version 2 of the License,  or  (at your
## option) any later version.
###########



import numpy as np
from scipy import fftpack
import scipy.ndimage.filters
import scipy.optimize as opt
import glob
from astropy.io import fits
import math
import matplotlib.pyplot as plt
import matplotlib


from .pistonrec import *
from .zernike import *
from smallfunc import *
from matplotlib import gridspec




##################################################
## Class 4: Functions to use Prex directly on yao data
##################################################

class Yaodata(Prex,Zernike):
    
    def __init__(self,path,prefix,size=0.5,crop=3,maxshift=4):
        # 'Global' parameters: path & prefix of Yao data,
        # size of fitted Gaussian,
        # size of cropped image for peak detection and 
        # maximal allowed shift in pixels
        self.path = path
        self.prefix = prefix
        self.size = size
        self.crop = crop
        self.maxshift = maxshift    
    
    
    def _yao2data(self,TT_subsystem=False,onedimslopes=False,full_TT=False,pyramid=False,leave_TT=False):
        """
        Loads Yao SH Data to do the prex algorithm or similar 
        The data has to be in the path folder, including:
    
        SH pupil:     prefix + '_SH_pupil.fits'
        Matrizes:     prefix + '-mat.fits'
        Slopes:       prefix + '_slopes_*.fits'
        DM Voltages:  prefix + '_dm1_alt_*.fits'
        
        returns: 2D SH sloeps (TT reduced) and tip & Tilt
        """
        # Read in WFS shape
        if pyramid:
            name_shshape = self.path + self.prefix + '_pyr_pupil_1.fits'
            shshape = fits.open(name_shshape)[0].data
            print('Using Pyramid shape of 1st WFS')
        else:
            try:
                name_shshape = self.path + self.prefix + '_SH_pupil_2.fits'
                shshape = fits.open(name_shshape)[0].data
                print('Using SH shape of 2nd SH-WFS')
            except:
                name_shshape = self.path + self.prefix + '_SH_pupil_1.fits'
                print('Only one SH-WFS, use shape of this')
                shshape = fits.open(name_shshape)[0].data            
        shshapelist = shshape.reshape(len(shshape)*len(shshape))
        validsubaps = np.sum(shshape)
        
        # How many DMs?
        dms = self.path + self.prefix + '_dm*_alt_0001.fits'
        number_dms = len(glob.glob(dms))
        print('%i DMs detected' % number_dms)
        
        if (number_dms > 2):
            raise Exception('Too many DMs or 2 DMs w/o a TT subsystem')
        if (number_dms == 2 and TT_subsystem == False):
            print('Two DMs detected, but no TT subsystem declared.')
            print('Will set TT_subsystem = True & Full_TT = True')
            TT_subsystem = True
            full_TT = True
            
        # Read in DM values
        if number_dms == 1:
            name_dm_val = self.path + self.prefix + '_dm1_alt_*.fits'
        if number_dms == 2:
            name_dm_val = self.path + self.prefix + '_dm2_alt_*.fits'
            name_dm_val_TT = self.path + self.prefix + '_dm1_alt_*.fits'
            filenames = sorted(glob.glob(name_dm_val_TT))
            dimt = len(filenames)
            hdudata = fits.open(filenames[0])
            dimx = hdudata[0].data.shape[0]
            supaps_TT_DM = dimx
            data_dm_TT = np.zeros((dimt,dimx))
            for index, filename in enumerate(filenames):
                hdudata = fits.open(filename)
                data_dm_TT[index] = hdudata[0].data.copy()
                del hdudata[0].data
                hdudata.close()
        filenames = sorted(glob.glob(name_dm_val))
        dimt = len(filenames)
        hdudata = fits.open(filenames[0])
        dimx = hdudata[0].data.shape[0]

        data_dm = np.zeros((dimt,dimx))
        for index, filename in enumerate(filenames):
            hdudata = fits.open(filename)
            data_dm[index] = hdudata[0].data.copy()
            del hdudata[0].data
            hdudata.close()
        
        # Read in CM and IM
        name_mat = self.path + self.prefix + '-mat.fits'
        mat = fits.open(name_mat)
        IM = mat[0].data[0]
        CM = mat[0].data[1]
        if TT_subsystem:
            name_shshape = self.path + self.prefix + '_SH_pupil_1.fits'
            shshape_TT = fits.open(name_shshape)[0].data            
            validsubaps_TT = np.sum(shshape_TT)
            IM_TT = IM[:supaps_TT_DM,:validsubaps_TT*2]
            IM = IM[supaps_TT_DM:,validsubaps_TT*2:]
            
        # Read in slope values
        name_slope_val = self.path + self.prefix + '_slopes_*.fits'
        filenames = sorted(glob.glob(name_slope_val))
        dimt = len(filenames)
        hdudata = fits.open(filenames[0])
        dimx = hdudata[0].data.shape[0]
    
        data_sl = np.zeros((dimt,dimx))
        for index, filename in enumerate(filenames):
            hdudata = fits.open(filename)
            data_sl[index] = hdudata[0].data.copy()
            del hdudata[0].data
            hdudata.close()
        if TT_subsystem:
            data_sl_TT = data_sl[:,:validsubaps_TT*2]
            data_sl = data_sl[:,validsubaps_TT*2:]

        # Create POL data
        dimx = data_sl.shape[1]
        data_pol = np.zeros((dimt-1,dimx))
        for i in range(dimt-1):
            data_pol[i] = data_sl[i+1]-np.dot(data_dm[i],IM)
            
        # check if several WFSs
        numWFS = (dimx//2)//validsubaps
        if numWFS > 1:
            print('Multiple WFSs, take the mean over the %i WFS' % numWFS)
            data_pol_long = np.copy(data_pol)
            data_pol = np.zeros((dimt-1,dimx//numWFS))
            for i in range(numWFS):
                data_pol += data_pol_long[:,i*2*validsubaps:i*2*validsubaps+2*validsubaps]
            data_pol /= numWFS       
            
        # Use the POL data to create 2D POL Data for x & y slopes
        data_pol_2D_x = np.zeros((dimt-1,shshape.shape[0],shshape.shape[1]))
        data_pol_2D_y = np.zeros((dimt-1,shshape.shape[0],shshape.shape[1]))
        tip = []
        tilt = []
        for i in range(dimt-1):
            k = 0
            data_pol_1D_x = np.zeros(len(shshapelist))
            data_pol_1D_y = np.zeros(len(shshapelist))
            for j in range(len(shshapelist)):
                if shshapelist[j] != 0:
                    data_pol_1D_x[j] = data_pol[i,k]
                    data_pol_1D_y[j] = data_pol[i,k+validsubaps]
                    k += 1
            data_pol_2D_x[i] = data_pol_1D_x.reshape(shshape.shape[0],shshape.shape[1])
            data_pol_2D_y[i] = data_pol_1D_y.reshape(shshape.shape[0],shshape.shape[1])
            mean_x = np.sum(data_pol_1D_x)/validsubaps
            tip.append(mean_x)
            mean_y = np.sum(data_pol_1D_y)/validsubaps
            tilt.append(mean_y)
            if leave_TT == False:
                data_pol_2D_x[i] -= mean_x
                data_pol_2D_y[i] -= mean_y
            data_pol_2D_x[i] *= shshape
            data_pol_2D_y[i] *= shshape    
        
        data_pol_1D_x = data_pol[:,:validsubaps]
        data_pol_1D_y = data_pol[:,validsubaps:]
        
        if TT_subsystem:
            tip_TT = []
            tilt_TT = []
            for i in range(1,dimt):
                pol_TT = data_sl_TT-np.dot(data_dm_TT[i-1],IM_TT)
                tip_TT.append(np.mean(pol_TT[i,:validsubaps_TT]))
                tilt_TT.append(np.mean(pol_TT[i,validsubaps_TT:]))
##            data_pol_TT = np.zeros((dimt-1,dimx))
##            for i in range(dimt-1):
##                data_pol_TT[i] = data_sl_TT[i+1]#-0.01*np.dot(data_dm_TT[i],IM_TT)
##            tip_TT = data_pol_TT[:,0]
##            tilt_TT = data_pol_TT[:,1]]
            if full_TT:
                print('Using tip, tilt from TT & GLAO system')
                tip = [i+j for i,j in zip(tip_TT,tip)]
                tilt = [i+j for i,j in zip(tilt_TT,tilt)]
            #plt.plot(tip2)
            #plt.plot(tilt2)
            #plt.show()
        
        datacube = [data_pol_2D_x,data_pol_2D_y,tip,tilt]
        
        if onedimslopes:
            return datacube, data_pol_1D_x, data_pol_1D_y
        else:
            return datacube
    
    
    
    
    
    def yao2prexTT(self,path_OL,average=10,npixel=200,return_lists=False,return_strehl=True,
                   use_piston=False,return_piston=False,calc_factor=False,plot=False,
                   tomum=0.19392,windaverage=1, *args, **kwargs):
        """
        Uses Yao SH Data to do the prex tip tilt algorithm
        The data has to be in the path folder, including:
    
        SH pupil:     prefix + '_SH_pupil.fits'
        Matrizes:     prefix + '-mat.fits'
        Slopes:       prefix + '_slopes_*.fits'
        DM Voltages:  prefix + '_dm1_alt_*.fits'
    
        Atmosphere:   'screen_l0_000_1.fits'
        OL Images:    '_rwf_*.fits'    -- Need an automatic solution
    
    
        Inputs:
        path_OL: path to folder with the yao fits files from the OL mode (string)
        return_lists:   if true returns two lists, the theoretical dif piston and the reconstructed
        use_piston:     If false a already calculated piston is used. Optional argument with keyword 'piston'
                        is needed.
                        As the piston calculation is the slowest part of the function, this option should be
                        used if there is a loop with the same atmosphere
        return_piston:  If true returns the theoretical piston values
        TT_subsystem:   for an GLAO I need an additional TT WFS in order to determine the theoretical values
                        then this has to be used with True
        
            
        Output:
        error:   rms error between differnetial piston from atmosphere and from calculation
        strehl:  measured strehl from yao (to compare the goodnes of the simulation run)
        piston:  list of reconstructed piston values (if return_piston)
        difpiston:  list of calculated differential piston (if return_lists)
        difpiston_rec:  list of reconstructed differential piston (if retun_lists)
        """

        datacube = self._yao2data(*args, **kwargs)
        data_pol_2D_x = datacube[0]
        data_pol_2D_y = datacube[1]
        tip = datacube[2]
        tilt = datacube[3]
        
        if plot:
            difpiston, maxx, maxy = self.prexTT(datacube,average,return_pos=True,windaverage=windaverage)
            # plot position
            z = np.arange(len(maxx))
            plt.figure(figsize=(4,4))
            plt.scatter(maxx,maxy,s=50,c=z,marker='o',zorder=3)
            plt.xlabel('X-Shift [pixel]')
            plt.ylabel('Y-Shift [pixel]')
            hide_spines(outwards=False)
            plt.show() 
        else:
            difpiston = self.prexTT(datacube,average,windaverage=windaverage)
        
        ## Calculate different factors
        #Number of lenslets in diameter for calculation facotr
        lenslets = len(data_pol_2D_x[0])
        fac = npixel/lenslets
        
        # Result of the funtion will now be in mu m
        # equavelnt to: (tel.diam/sim.pupildiam)/4.848
        # for D = 8m, 200 pix: factor = 0.19393
        if tomum == 0.19392:
            print('Standard Factor for D=8m & 200 pixel used. Need to be changed for a different konfiguration.')
            print('Use keyword factor')
        else:
            print('Calculation factor used: %.5f' % tomum)
        
        difpiston = [i*fac*tomum for i in difpiston]


        if use_piston and ('piston' in kwargs):
            piston = kwargs['piston']
        
        else:
            print('\nCalculate the theoretical piston values. This will take some time and')
            print('should be avoided if possible \n')    
            if use_piston and ('piston' not in kwargs):
                print('No Piston values given, have to calculate them.')
            # Import OL data
            # different path, final solution should be with same path
            name_ol_val = path_OL + self.prefix + '_rwf_*.fits'
            filenames = sorted(glob.glob(name_ol_val))
            dimt = len(filenames)
            hdudata = fits.open(filenames[0])[0].data
            pixel = np.count_nonzero(hdudata[len(hdudata)//2,:])
            dif = (len(hdudata)-pixel)//2
            boundary = int(math.ceil(pixel/2-pixel/(2*math.sqrt(2))))
        
            data_OL = np.zeros((dimt,len(hdudata)-2*dif,len(hdudata)-2*dif))
            for index, filename in enumerate(filenames):
                hdudata = fits.open(filename)
                data = hdudata[0].data.copy()
                del hdudata[0].data
                hdudata.close()
                data_OL[index] = self.mask(data[dif:-dif,dif:-dif],mask_val=np.nan)
                data_OL[index] -= np.nanmean(data_OL[index])
                
            # Get Piston and shift from used atmosphere
            # Does not use mid pixel shift yet
            # Is there a faster solution? not calculating every piston? only the one I use?
            name_atmosphere = self.path + 'screen1.fits'
            atmosphere = fits.open(name_atmosphere)[0].data
        
            maxx = []
            maxy = []
            for i in range(len(data_OL)):
                print_status(i,len(data_OL))
                phase1 = data_OL[i][boundary:-boundary,boundary:-boundary]
                nxcorr = self.nxcorrelation(atmosphere,phase1)
                maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
                maxx.append(maxpos[1])
                maxy.append(maxpos[0])
            
            piston = []
            for i in range(len(maxx)):
                image = np.zeros((pixel,pixel))
                image = np.copy(atmosphere[maxy[i]-pixel//2:maxy[i]+pixel//2,maxx[i]-pixel//2:maxx[i]+pixel//2])
                image = self.mask(image,mask_val=np.nan)
                piston.append(np.nanmean(image))
        
        # weighting factor rad -> mum, incl correct D/r0 & airmass
        piston_weight = np.genfromtxt(self.path + self.prefix + '_weight')
        if len(piston_weight) > 1:
            piston_weight = piston_weight[0]
        difpiston_rec = []
        for j in range(0,len(tip)-average,average):
            dif = (np.mean(piston[j+average:j+2*average])-np.mean(piston[j:j+average]))
            difpiston_rec.append(dif*piston_weight)
        
        
        if plot:
            plt.figure(figsize=(6,3))
            plt.plot(difpiston,color=color1,ls='',marker='o',label='measured dif. piston')
            plt.plot(difpiston_rec,color='k',label='recovered dif.piston')
            plt.axhline(0,lw=0.4)
            hide_spines()
            plt.legend(loc=2)
            plt.xlabel('Measurements')
            plt.ylabel('dPiston [$\mu$m]')
            plt.show()
        print(difpiston)
        print(difpiston_rec)
        error = rmse(difpiston,difpiston_rec)
        
        if return_strehl:
            name_strehl = self.path + self.prefix + '_strehl'
            strehl = float(np.genfromtxt(name_strehl))
        else:
            strehl = 0.0
        
        
        if return_lists:
            if return_piston:
                return error, strehl, piston, difpiston, difpiston_rec
            else:
                return error, strehl, difpiston, difpiston_rec
        else:
            if return_piston:
                return error, strehl, piston
            else:
                return error, strehl










    def yao2prexTTfast(self,average=10,tomum=0.19392,npixel=200,return_piston=False,plot=False,
                    return_lists=False,return_strehl=True,TTfactor=False,windaverage=1,
                    *args, **kwargs):
        """
        Uses Yao SH Data to do the prex tip tilt algorithm
        The data has to be in the path folder, including:
        
        SH pupil:     prefix + '_SH_pupil.fits'
        Matrizes:     prefix + '-mat.fits'
        Slopes:       prefix + '_slopes_*.fits'
        DM Voltages:  prefix + '_dm1_alt_*.fits'
        Input phases: prefix + '_weight_*.fits'
        
        Atmosphere:   'screen_l0_000_1.fits'    
        
        Inputs:
        path:    path to folder with the yao fits files (string)
        path_OL: path to folder with the yao fits files from the OL mode (string)
        prefix:  prefix of the yao par file (string)
        average: Number of measurements to use for the average, number around 10 seems to be reasonable
        size:    size of the Gaussian which will be fitted. If the results are bad, this can be a possibility 
                for a change
                
        return_lists:   if true returns two lists, the theoretical dif piston and the reconstructed
        use_piston:     If false a already calculated piston is used. Optional argument with keyword 'piston'
                        is needed.
                        As the piston calculation is the slowest part of the function, this option should be
                        used if there is a loop with the same atmosphere
        return_piston:  If true returns the theoretical piston values
        TT_subsystem:   for an GLAO I need an additional TT WFS in order to determine the theoretical values
                        then this has to be used with True
                
                
        nearly the same as 'yao_to_prex_TT' but it gets the theoretical piston from the input phase (used by YAO)
        which avoids a large cross correlatino and makes the computation much faster
                
        Output:
        error:   rms error between differnetial piston from atmosphere and from calculation
        strehl:  measured strehl from yao (to compare the goodnes of the simulation run)
        piston:  list of reconstructed piston values (if return_piston)
        difpiston:  list of calculated differential piston (if return_lists)
        difpiston_rec:  list of reconstructed differential piston (if retun_lists)
        """
        datacube = self._yao2data(*args, **kwargs)
        data_pol_2D_x = datacube[0]
        data_pol_2D_y = datacube[1]
        tip = datacube[2]
        tilt = datacube[3]


        ## Calculate different factors
        #Number of lenslets in diameter for calculation facotr
        lenslets = len(data_pol_2D_x[0])
        fac = npixel/lenslets
        
        # Result of the funtion will now be in mu m
        # equavelnt to: (tel.diam/sim.pupildiam)/4.848
        # for D = 8m, 200 pix: factor = 0.19393
        if tomum == 0.19392:
            print('Standard Factor for D=8m & 200 pixel used. Need to be changed for a different konfiguration.')
            print('Use keyword factor')
        else:
            print('Calculation factor used: %.5f' % tomum)
            
        tip = [i*tomum for i in tip]
        tilt = [i*tomum for i in tilt]
        
        # all data to do prexTT
        datacube = [data_pol_2D_x,data_pol_2D_y,tip,tilt]
        
        if plot:
            difpiston, maxx, maxy = self.prexTT(datacube,average,return_pos=True,windaverage=windaverage)
            # plot position
            z = np.arange(len(maxx))
            plt.figure(figsize=(10,3))
            gs = gridspec.GridSpec(1, 5,width_ratios=(1,0.1,1,0.1,1))
            axes = plt.subplot(gs[0,0])
            plt.scatter(maxx,maxy,s=50,c=z,marker='o',zorder=3)
            #plt.xlabel('X-Shift [pixel]')
            #plt.ylabel('Y-Shift [pixel]')
            hide_spines(outwards=False)
            print('X-Shift: %.3f +- %.3f' % (np.mean(maxx), np.std(maxx)))
            print('Y-Shift: %.3f +- %.3f' % (np.mean(maxy), np.std(maxy)))
        elif return_lists:
            difpiston, maxx, maxy = self.prexTT(datacube,average,return_pos=True,windaverage=windaverage)
        else:
            difpiston = self.prexTT(datacube,average,windaverage=windaverage)

        
        # Theoretical Values
        name_weight_val = self.path + self.prefix + '_weight_*.fits'
        filenames = sorted(glob.glob(name_weight_val))
        if TTfactor or plot:
            tip_theo = []
            tilt_theo = []
        piston = []
        for idx, filename in enumerate(filenames):
            hdudata = fits.open(filename)
            data = hdudata[0].data[1:-1,1:-1].copy()
            data = self.mask(data,mask_val=np.nan)[1:-1,1:-1]
            if idx == 0:
                npixel = len(data)
            piston.append(np.nanmean(data))
            if TTfactor or plot:
                tilt_theo.append(np.nanmean(np.gradient(data,axis=0)))
                tip_theo.append(np.nanmean(np.gradient(data,axis=1)))
        
            del hdudata[0].data
            hdudata.close()
            
        difpiston_rec = []
        for j in range(0,len(tip)-average,average):
            dif = (np.mean(piston[j+average:j+2*average])-np.mean(piston[j:j+average]))
            difpiston_rec.append(dif)
# Not needed with new wind average calculation
#        if windaverage != 1:
#            overlap = (len(difpiston_rec)%windaverage)
#            difpiston_rec = difpiston_rec[:-overlap]
        
        # if wanted get calibration factor for TT values
        if TTfactor:
            tip_theo2 = []
            tilt_theo2 = []
            av_tip_list = []
            av_tilt_list = []
            for i in range(0,len(piston)-average,average):
                tip_theo2.append(np.mean(tip_theo[i:i+2*average]))
                tilt_theo2.append(np.mean(tilt_theo[i:i+2*average]))
                av_tip_list.append(np.mean(tip[i:i+2*average]))
                av_tilt_list.append(np.mean(tilt[i:i+2*average]))
            
            mtip_theo = np.mean(tip_theo2)
            mtilt_theo = np.mean(tilt_theo2)
            factor_tip = mtip_theo/np.mean(av_tip_list)
            factor_tilt = mtilt_theo/np.mean(av_tilt_list)
            factorTT = 0.5*(factor_tip+factor_tilt)
            difpiston = [i*factorTT for i in difpiston]
            if plot:
                av_tip_list = [i*factorTT for i in av_tip_list]
                av_tilt_list = [i*factorTT for i in av_tilt_list]
            print('TT factor of %.3f used' % factorTT)
#            return factorTT, tip_theo2, tilt_theo2, av_tip_list,av_tilt_list
        elif plot:
            tip_theo2 = []
            tilt_theo2 = []
            av_tip_list = []
            av_tilt_list = []
            for i in range(0,len(piston)-average,average):
                tip_theo2.append(np.mean(tip_theo[i:i+2*average]))
                tilt_theo2.append(np.mean(tilt_theo[i:i+2*average]))
                av_tip_list.append(np.mean(tip[i:i+2*average]))
                av_tilt_list.append(np.mean(tilt[i:i+2*average]))
        
        difpiston = [i*fac for i in difpiston]
        
        
        if plot:
            axes = plt.subplot(gs[0,2])
            plt.plot(av_tip_list,color=color1,marker='',ls='-',label='Measured Tip')
            plt.plot(tip_theo2,color=color1,marker='.',ls='',label='Theoretical Tip')
            plt.plot(av_tilt_list,color=color3,marker='',ls='-',label='Measured Tilt')
            plt.plot(tilt_theo2,color=color3,marker='.',ls='',label='Theoretical Tilt')
            hide_spines()
            #plt.legend(loc=2)
            #plt.xlabel('Measurements')
            #plt.ylabel('TipTilt')
            
            
            axes = plt.subplot(gs[0,4])
            plt.plot(difpiston,color=color1,ls='',marker='o',label='measured dif. piston')
            plt.plot(difpiston_rec,color='k',label='recovered dif.piston')
            plt.axhline(0,lw=0.4)
            hide_spines()
            #plt.legend(loc=2)
            #plt.xlabel('Measurements')
            #plt.ylabel('dPiston [$\mu$m]')
            plt.show()
    
        

        
        error = rmse(difpiston,difpiston_rec)
        
        if return_strehl:
            name_strehl = self.path + self.prefix + '_strehl'
            strehl = float(np.genfromtxt(name_strehl))
            print('Strehl: %.3f' % strehl)
        else:
            strehl = 0.0
        
        if return_piston:
            if return_lists:
                return error, strehl, difpiston, difpiston_rec, piston, maxx, maxy
            else:
                return error, strehl, piston

        else:
            if return_lists:
                return error, strehl, difpiston, difpiston_rec, maxx, maxy
            else:
                return error, strehl




    def yao2prexTTpiston(self,average=10,frequency=500,only_rms=False,fast=True, plot_piston=False,
                         windaverage=1,*args, **kwargs):
        """
        Uses the yao_to_prex_TT_fast function (for necessary data see function) 
        but gives the reconstructed piston
        as an output and not the differential piston
        
        outputs:
        x           x scale in seconds
        piston_red  theoretical piston (reduced), first value set to zero
        piston_rec  recovered piston
        piston_res  residual between reduced and recovered piston
        piston_rms  rms of residual piston
        """
        if fast:
            prexdata = self.yao2prexTTfast(average=average,windaverage=windaverage,return_piston=True,
                                           return_lists=True, *args, **kwargs)
        else:
            prexdata = self.yao2prexTT(self.path,average=average,return_piston=True,
                                       return_lists=True, *args, **kwargs)
            
        
        piston = prexdata[4]
        pistonred = [i-piston[0] for i in piston]
        piston_red = pistonred[0:-average:average]
        
        piston_rec = []
        for i in range(len(prexdata[2])):
            piston_rec.append(sum(prexdata[2][:i]))

        piston_res = []
        for i in range(len(prexdata[2])):
            pis = sum(prexdata[2][:i])
            piston_res.append(piston_red[i]-pis)

        piston_rms = rms(piston_res)
        
        piston_red = piston_red[:len(piston_rec)]
        
        step = average/frequency
        x = np.arange(len(piston_rec))*step
           
        if plot_piston:
            try:
                plt.plot(x,piston_rec,color=color1,label='Reconstructed piston')
                plt.plot(x,piston_red,color='k',label='Theoretical piston')
                plt.plot(x,piston_res,color=color3,label='Residual Value')
                plt.fill_between(x,piston_res, 0, alpha=0.2,color=color3)
                plt.axhline(0,lw=0.4)
                hide_spines()
                plt.legend(loc=2)
                plt.xlabel('Time [s]')
                plt.ylabel('Piston [$\mu$m]')
                plt.show()
            except ValueError:
                plt.clf()
                plt.plot(piston_rec,color=color1,label='Reconstructed piston')
                plt.plot(piston_red,color='k',label='Theoretical piston')
                plt.plot(piston_res,color=color2,label='Residual Value')
                plt.fill_between(piston_res, 0, alpha=0.2,color=color2)
                plt.axhline(0,lw=0.4)
                hide_spines()
                plt.legend(loc=2)
                plt.xlabel('measurements')
                plt.ylabel('Piston [$\mu$m]')
                plt.show()
        
        if only_rms:
            return piston_rms
        else:
            return x, piston_red, piston_rec, piston_res, piston_rms
        
        
        
    
    #############################
    ## 2nd prex version, with WF reconstruction
    ## usually not used, as it is way slower, and less
    ## accurate
    #############################
    
    
    def yao2prex(self,path_OL,average=10,tomum=0.19392,npixel=200,
                return_lists=False,return_strehl=True, use_piston=False,
                return_piston=False, *args, **kwargs):
        """
        Uses Yao SH Data to do the prex algorithm
        The data has to be in the path folder, including:
        
        SH pupil:     prefix + '_SH_pupil.fits'
        Matrizes:     prefix + '-mat.fits'
        Slopes:       prefix + '_slopes_*.fits'
        DM Voltages:  prefix + '_dm1_alt_*.fits'
        
        Atmosphere:   'screen_l0_000_1.fits'
        OL Images:    '_rwf_*.fits'    -- Need an automatic solution
        
        
        Inputs:
        path:    path to folder with the yao fits files (string)
        path_OL: path to folder with the yao fits files from the OL mode (string)
        prefix:  prefix of the yao par file (string)
        average: Number of measurements to use for the average, number around 10 seems to be reasonable
        size:    size of the Gaussian which will be fitted. If the results are bad, this can be a possibility 
                for a change
                
        return_lists:   if true returns two lists, the theoretical dif piston and the reconstructed
        use_piston:     If false a already calculated piston is used. Optional argument with keyword 'piston'
                        is needed.
                        As the piston calculation is the slowest part of the function, this option should be
                        used if there is a loop with the same atmosphere
        return_piston:  If true returns the theoretical piston values
                
        
        The function uses the function "recon_WF_from_slopes" to reconstruct the encessary wavefront from the
        measured slopes
        At the moment it also uses "build_Recon_Matrix" every time, this could be solved more intelligent
        to avoid the computation every time (not super slow, but unneccesarry)
        
        
        Output:
        error:   rms error between differnetial piston from atmosphere and from calculation
        strehl:  measured strehl from yao (to compare the goodnes of the simulation run)
        piston:  list of reconstructed piston values (if return_piston)
        difpiston:  list of calculated differential piston (if return_lists)
        difpiston_rec:  list of reconstructed differential piston (if retun_lists)
        """
        datacube, data_pol_1D_x, data_pol_1D_y = self._yao2data(onedimslopes=True)
        data_pol_2D_x = datacube[0]
        data_pol_2D_y = datacube[1]
        tip = datacube[2]
        tilt = datacube[3]


        ## Calculate different factors
        #Number of lenslets in diameter for calculation facotr
        lenslets = len(data_pol_2D_x[0])
        fac = npixel/lenslets
    
    
        # Get the wind vecor
        maxx, maxy = self.prexTT(datacube,average,only_pos=True)


        # Read in SH shape
        name_shshape = self.path + self.prefix + '_SH_pupil.fits'
        shshape = fits.open(name_shshape)[0].data
        shshapelist = shshape.reshape(len(shshape)*len(shshape))
        validsubaps = np.sum(shshape)
        nlensletsdiam = len(shshape)
    
        # Get the differential psiton
        RM = self._zernreconstruction(validsubaps,100,nlensletsdiam,returnMask=False)    
        difpiston = []
        k = 0
        print('Doing P-REx:')
        for i in range(0,len(data_pol_1D_x)-average,average):
            print_status(i,len(data_pol_1D_x)-average)
            
            vx = np.int(np.round(maxx[k]*fac))
            vy = np.int(np.round(maxy[k]*fac))

            rwavefront0 = self.slopes2WF(RM,data_pol_1D_x[i],data_pol_1D_y[i],shshape)
            rwavefront0 -= np.nanmean(rwavefront0)
            rwavefront1 = self.slopes2WF(RM,data_pol_1D_x[i+average],data_pol_1D_y[i+average],shshape)
            rwavefront1 -= np.nanmean(rwavefront1)
        
            # Not sure if everything is correct, needs testing!
            phase0 = np.empty((rwavefront0.shape[0]+np.abs(vx),rwavefront0.shape[1]+np.abs(vy)))
            phase0[:] = np.nan
            phase1 = np.empty((rwavefront1.shape[0]+np.abs(vx),rwavefront1.shape[1]+np.abs(vy)))
            phase1[:] = np.nan
            
            
            if vx > 0:
                if vy > 0:
                    phase0[:-vx,:-vy] = rwavefront0
                    phase1[vx:,vy:] = rwavefront1
                elif vy < 0:
                    phase0[:-vx,-vy:] = rwavefront0
                    phase1[vx:,:vy] = rwavefront1
                else:
                    phase0[:-vx,:] = rwavefront0
                    phase1[vx:,:] = rwavefront1
                
            elif vx < 0:
                if vy > 0:
                    phase0[-vx:,:-vy] = rwavefront0
                    phase1[:vx,vy:] = rwavefront1
                elif vy < 0:
                    phase0[-vx:,-vy:] = rwavefront0
                    phase1[:vx,:vy] = rwavefront1
                else:
                    phase0[-vx:,:] = rwavefront0
                    phase1[:vx,:] = rwavefront1
                
            else:
                if vy > 0:
                    phase0[:,:-vy] = rwavefront0
                    phase1[:,vy:] = rwavefront1
                elif vy < 0:
                    phase0[:,-vy:] = rwavefront0
                    phase1[:,:vy] = rwavefront1       
                else:
                    phase0 = rwavefront0
                    phase1 = rwavefront1
        
            bigmask = np.ones_like(phase1)
            bigmask[np.where(np.isnan(phase0*phase1))] = np.nan
            
            area0 = bigmask*phase0
            area1 = bigmask*phase1   
            
            difpiston.append(np.nanmean(area0) - np.nanmean(area1))
            k += 1
        
        # Result of the funtion will now be in mu m
        # equavelnt to: (tel.diam/sim.pupildiam)/4.848
        # for D = 8m, 200 pix: factor = 0.19393
        if tomum == 0.19392:
            print('\nStandard Factor for D=8m & 200 pixel used. Need to be changed for a different konfiguration.')
            print('Use keyword factor')
        else:
            print('\nCalculation factor used: %.5f' % tomum)
        
        difpiston = [i*tomum*110 for i in difpiston]

    
        if use_piston and ('piston' in kwargs):
            piston = kwargs['piston']
        
        else:
            print('\n Calculate the theoretical piston values. This will take some time')
            print('Should be avoided if possible \n')    
            if use_piston and ('piston' not in kwargs):
                print('No Piston values given, have to calculate them.')
            # Import OL data
            # different path, final solution should be with same path
            name_ol_val = path_OL + self.prefix + '_rwf_*.fits'
            filenames = sorted(glob.glob(name_ol_val))
            dimt = len(filenames)
            hdudata = fits.open(filenames[0])[0].data
            pixel = np.count_nonzero(hdudata[len(hdudata)//2,:])
            dif = (len(hdudata)-pixel)//2
            boundary = int(math.ceil(pixel/2-pixel/(2*math.sqrt(2))))
        
            data_OL = np.zeros((dimt,len(hdudata)-2*dif,len(hdudata)-2*dif))
            for index, filename in enumerate(filenames):
                hdudata = fits.open(filename)
                data = hdudata[0].data.copy()
                del hdudata[0].data
                hdudata.close()
                data_OL[index] = self.mask(data[dif:-dif,dif:-dif],mask_val=np.nan)
                data_OL[index] -= np.nanmean(data_OL[index])
        
            # Get Piston and shift from used atmosphere
            # Does not use mid pixel shift yet
            # Is there a faster solution? not calculating every piston? only the one I use?
            name_atmosphere = self.path + 'screen1.fits'
            atmosphere = fits.open(name_atmosphere)[0].data
        
            maxx = []
            maxy = []
            for i in range(len(data_OL)):
                print_status(i,len(data_OL))
                phase1 = data_OL[i][boundary:-boundary,boundary:-boundary]
                nxcorr = self.nxcorrelation(atmosphere,phase1)
                maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
                maxx.append(maxpos[1])
                maxy.append(maxpos[0])
            
            piston = []
            for i in range(len(maxx)):
                image = np.zeros((pixel,pixel))
                image = np.copy(atmosphere[maxy[i]-pixel//2:maxy[i]+pixel//2,maxx[i]-pixel//2:maxx[i]+pixel//2])
                image = self.mask(image,mask_val=np.nan)
                piston.append(np.nanmean(image))
            print('\nDone. \n')
    
    
        # weighting factor rad -> mum, incl correct D/r0 & airmass
        piston_weight = np.genfromtxt(self.path + self.prefix + '_weight')
        
        difpiston_rec = []
        for j in range(0,len(tip)-average,average):
            dif = (np.mean(piston[j+average:j+2*average])-np.mean(piston[j:j+average]))
            difpiston_rec.append(dif*piston_weight)
        
        error = rmse(difpiston,difpiston_rec)
        
        if return_strehl:
            name_strehl = self.path + self.prefix + '_strehl'
            strehl = float(np.genfromtxt(name_strehl))
        else:
            strehl = 0.0
            
            
        if return_lists:
            if return_piston:
                return error, strehl, piston, difpiston, difpiston_rec
            else:
                return error, strehl, difpiston, difpiston_rec
        else:
            if return_piston:
                return error, strehl, piston
            else:
                return error, strehl    




    
    
    
    
    
    
    
    
    
    def yao2prexfast(self,average=10,tomum=0.19392,npixel=200,return_piston=False,
                    return_lists=False,return_strehl=True,TTfactor=False, *args, **kwargs):
        """
        Uses Yao SH Data to do the prex algorithm (publication will follow)
        The data has to be in the path folder, including:
        
        SH pupil:     prefix + '_SH_pupil.fits'
        Matrizes:     prefix + '-mat.fits'
        Slopes:       prefix + '_slopes_*.fits'
        DM Voltages:  prefix + '_dm1_alt_*.fits'
        
        Atmosphere:   'screen_l0_000_1.fits'
        OL Images:    '_rwf_*.fits'    -- Need an automatic solution
        
        
        Inputs:
        average: Number of measurements to use for the average, number around 10 seems to be reasonable
        return_lists:   if true returns two lists, the theoretical dif piston and the reconstructed
        return_piston:  If true returns the theoretical piston values
                
        
        The function uses the function "slopes2WF" to reconstruct the necessary wavefront from the
        measured slopes
        At the moment it also uses "zernreconstruction" every time, this could be solved more intelligent
        to avoid the computation every time (not super slow, but unneccesarry)
        
        Output:
        error:   rms error between differnetial piston from atmosphere and from calculation
        strehl:  measured strehl from yao (to compare the goodnes of the simulation run)
        piston:  list of reconstructed piston values (if return_piston)
        difpiston:  list of calculated differential piston (if return_lists)
        difpiston_rec:  list of reconstructed differential piston (if retun_lists)
        """
        datacube, data_pol_1D_x, data_pol_1D_y = self._yao2data(onedimslopes=True)
        data_pol_2D_x = datacube[0]
        data_pol_2D_y = datacube[1]
        tip = datacube[2]
        tilt = datacube[3]


        ## Calculate different factors
        #Number of lenslets in diameter for calculation facotr
        lenslets = len(data_pol_2D_x[0])
        fac = npixel/lenslets
    
        # Get the wind vecor
        maxx, maxy = self.prexTT(datacube,average,only_pos=True)


        # Read in SH shape
        name_shshape = self.path + self.prefix + '_SH_pupil_1.fits'
        shshape = fits.open(name_shshape)[0].data
        shshapelist = shshape.reshape(len(shshape)*len(shshape))
        validsubaps = np.sum(shshape)
        nlensletsdiam = len(shshape)
    
        # Get the differential psiton
        RM = self._zernreconstruction(validsubaps,100,nlensletsdiam,returnMask=False)    
        difpiston = []
        k = 0
        print('Doing P-REx:')
        for i in range(0,len(data_pol_1D_x)-average,average):
            print_status(i,len(data_pol_1D_x)-average)
            
            vx = np.int(np.round(maxx[k]*fac))
            vy = np.int(np.round(maxy[k]*fac))

            rwavefront0 = self.slopes2WF(RM,data_pol_1D_x[i],data_pol_1D_y[i],shshape)
            rwavefront0 -= np.nanmean(rwavefront0)
            rwavefront1 = self.slopes2WF(RM,data_pol_1D_x[i+average],data_pol_1D_y[i+average],shshape)
            rwavefront1 -= np.nanmean(rwavefront1)
        
            # Not sure if everything is correct, needs testing!
            phase0 = np.empty((rwavefront0.shape[0]+np.abs(vx),rwavefront0.shape[1]+np.abs(vy)))
            phase0[:] = np.nan
            phase1 = np.empty((rwavefront1.shape[0]+np.abs(vx),rwavefront1.shape[1]+np.abs(vy)))
            phase1[:] = np.nan
            
            
            if vx > 0:
                if vy > 0:
                    phase0[:-vx,:-vy] = rwavefront0
                    phase1[vx:,vy:] = rwavefront1
                elif vy < 0:
                    phase0[:-vx,-vy:] = rwavefront0
                    phase1[vx:,:vy] = rwavefront1
                else:
                    phase0[:-vx,:] = rwavefront0
                    phase1[vx:,:] = rwavefront1
                
            elif vx < 0:
                if vy > 0:
                    phase0[-vx:,:-vy] = rwavefront0
                    phase1[:vx,vy:] = rwavefront1
                elif vy < 0:
                    phase0[-vx:,-vy:] = rwavefront0
                    phase1[:vx,:vy] = rwavefront1
                else:
                    phase0[-vx:,:] = rwavefront0
                    phase1[:vx,:] = rwavefront1
                
            else:
                if vy > 0:
                    phase0[:,:-vy] = rwavefront0
                    phase1[:,vy:] = rwavefront1
                elif vy < 0:
                    phase0[:,-vy:] = rwavefront0
                    phase1[:,:vy] = rwavefront1       
                else:
                    phase0 = rwavefront0
                    phase1 = rwavefront1
        
            bigmask = np.ones_like(phase1)
            bigmask[np.where(np.isnan(phase0*phase1))] = np.nan
            
            area0 = bigmask*phase0
            area1 = bigmask*phase1   
            
            difpiston.append(np.nanmean(area0) - np.nanmean(area1))
            k += 1
        
        # Result of the funtion will now be in mu m
        # equavelnt to: (tel.diam/sim.pupildiam)/4.848
        # for D = 8m, 200 pix: factor = 0.19393
        if tomum == 0.19392:
            print('\nStandard Factor for D=8m & 200 pixel used. Need to be changed for a different konfiguration.')
            print('Use keyword factor')
        else:
            print('\nCalculation factor used: %.5f' % tomum)
        
        difpiston = [i*tomum*110 for i in difpiston]


        # Get weight and piston
        name_weight_val = self.path + self.prefix + '_weight_*.fits'
        filenames = sorted(glob.glob(name_weight_val))
        if TTfactor:
            tip_theo = []
            tilt_theo = []
        piston = []
        for filename in filenames:
            hdudata = fits.open(filename)
            data = hdudata[0].data[1:-1,1:-1].copy()
            data = self.mask(data,mask_val=np.nan)[1:-1,1:-1]
            piston.append(np.nanmean(data))
            if TTfactor:
                tipmean = np.nanmean(data,0)
                tiltmean = np.nanmean(data,1)
                tilt_theo.append(np.nanmean(np.gradient(tiltmean)))
                tip_theo.append(np.nanmean(np.gradient(tipmean)))
            del hdudata[0].data
            hdudata.close()
        
            

        if TTfactor:
            av_tip = []
            av_tilt = []
            tip_theo2 = []
            tilt_theo2 = []
            for i in range(0,len(tip)-average,average):
                av_tip.append(np.mean(tip[i:i+2*average])*0.19392)
                av_tilt.append(np.mean(tilt[i:i+2*average])*0.19392)
                tip_theo2.append(np.mean(tip_theo[i:i+2*average]))
                tilt_theo2.append(np.mean(tilt_theo[i:i+2*average]))   
            factor_tip = [i/j for i,j in zip(tip_theo2,av_tip)]
            factor_tip = np.mean(factor_tip)
            factor_tilt = [i/j for i,j in zip(tilt_theo2,av_tilt)]
            factor_tilt = np.mean(factor_tilt)
            factorTT = (factor_tilt+factor_tip)/2
            difpiston = [i*factorTT for i in difpiston]
            print('TT factor of %.3f used' % factorTT)
        
        
        
        difpiston_rec = []
        for j in range(0,len(tip)-average,average):
            dif = (np.mean(piston[j+average:j+2*average])-np.mean(piston[j:j+average]))
            difpiston_rec.append(dif)
        
        error = rmse(difpiston,difpiston_rec)
        
        if return_strehl:
            name_strehl = self.path + self.prefix + '_strehl'
            strehl = float(np.genfromtxt(name_strehl))
        else:
            strehl = 0.0
            
            
        if return_piston:
            if return_lists:
                return error, strehl, difpiston, difpiston_rec, piston
            else:
                return error, strehl, piston

        else:
            if return_lists:
                return error, strehl, difpiston, difpiston_rec
            else:
                return error, strehl





    
    def yao2prexpiston(self,average=10,frequency=500,only_rms=False,plot_piston=False,TTfactor=False, *args, **kwargs):
        """
        Uses the yao_to_prex_fast function (for necessary data see function) but gives the reconstructed piston
        as an output and not the differential piston
        
        outputs:
        x           x scale in seconds
        piston_red  theoretical piston (reduced), first value set to zero
        piston_rec  recovered piston
        piston_res  residual between reduced and recovered piston
        piston_rms  rms of residual piston
        """
        data_prex = self.yao2prexfast(average=average,return_piston=True,return_lists=True,
                                      TTfactor=TTfactor, *args, **kwargs)
    
        piston = data_prex[4]
        pistonred = [i-piston[0] for i in piston]
        piston_red = pistonred[0:-average:average]
        
        piston_rec = []
        for i in range(len(data_prex[2])):
            piston_rec.append(sum(data_prex[2][:i]))

        piston_res = []
        for i in range(len(data_prex[2])):
            pis = sum(data_prex[2][:i])
            piston_res.append(piston_red[i]-pis)

        piston_rms = rms(piston_res)
        
        step = average/frequency
        x = np.arange(0,len(data_prex[2])*step,step)
        if plot_piston:
            plt.figure(figsize=(5,2.5))
            plt.plot(x,piston_rec,color=color1,label='Reconstructed piston')
            plt.plot(x,piston_red,color='k',label='Theoretical piston')
            plt.plot(x,piston_res,color=color2,label='Residual Value')
            plt.fill_between(x,piston_res, 0, alpha=0.2,color=color2)
            plt.axhline(0,lw=0.4)
            hide_spines()
            plt.legend(loc=2)
            plt.xlabel('Time [s]')
            plt.ylabel('Piston [$\mu$m]')
            plt.show()
            
        if only_rms:
            return piston_rms
        else:
            return x, piston_red, piston_rec, piston_res, piston_rms
    
                

    
    
