###########
## This file is part of the Python Module P-Rex, a module for test for a 
## piston reconstruction experiement for optical interferometer (see Pott et al 2016)
## This files contains several functions to use the idea of piston reconstruction on 
## AO data
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

from .imagepro import *


##################################################
## Class 3: P-REx functions
##################################################

class Prex(Imagepro):
    
    def __init__(self,size=0.5,crop=3,maxshift=4):
        self.size = size
        self.crop = crop
        self.maxshift = maxshift
    
    
    def winddetection(self,image,kernel,laplace=False,return_image=False):
        """
        Get shift between to images with a normalized cross correlation and a fitted Gaussian
        Similar functino later used in the prex system
        """
        nxcorr = self.nxcorrelation(image,kernel,laplace=laplace)
    
        try:
            maxpos = self.maxgauss(nxcorr,crop=self.crop,size=self.size)
        except (RuntimeError, TypeError):
            print("Error - curve_fit failed; try with different size")
            try:
                maxpos = self.maxgauss(nxcorr,crop=self.crop,size=self.size)
            except (RuntimeError, TypeError):
                    maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
                    print("Error - curve_fit failed, use maxpos")
        x,y = [j-len(nxcorr)//2 for j in maxpos]
        
        if (abs(x)>self.maxshift) or (abs(y)>self.maxshift):
            try:
                maxpos = self.maxgauss(nxcorr,crop=self.crop,size=self.size*2)
                x,y = [j-len(nxcorr)//2 for j in maxpos]
            except (RuntimeError, TypeError):
                print('Value for shift seems unreasonable, second try failed')
            if (abs(x)>self.maxshift) or (abs(y)>self.maxshift):
                print('Value for shift seems unreasonable')
        
        if return_image:
            return x, y, nxcorr
        return x, y
    
    

    
    
    
    
    def prexTT(self,datacube,average,return_pos=False,only_pos=False):
        """
        Function to calculate the differential piston from slope data using the
        Tip-Tilt concept (documentation/paper in preparation)
        
        input data:
        datacube: list with 2D xslopes data, 2D yslopes, tip and tilt
        average: number of images for the average, measurement will be for 2*average
        return_pos: if true returns also list of the posisiton of the maximum
        only_pos: if True returns only the list of the posisiton of the maximum (wind vector)
        !!! Calibration factor might be necessary !!!
        !!! More elegant to do a slope2shift function which is then called here !!!
        """
        
        if len(datacube) != 4:
            raise Exception('Input data has to be a list with 4 entries (x & y slopes, tip & tilt)')
        
        xslopes = datacube[0]
        yslopes = datacube[1]
        tip = datacube[2]
        tilt = datacube[3]
        
        difpiston = []

        maxx = []
        maxy =[]
        for i in range(0,len(tip)-average,average):
            image_x = np.mean(xslopes[i:i+average],axis=0)
            kernel_x = np.mean(xslopes[i+average:i+2*average],axis=0)
            nxcorr_x = self.nxcorrelation(image_x,kernel_x,laplace=False)
    
            image_y = np.mean(yslopes[i:i+average],axis=0)
            kernel_y = np.mean(yslopes[i+average:i+2*average],axis=0)
            nxcorr_y = self.nxcorrelation(image_y,kernel_y,laplace=False)
            nxcorr = (nxcorr_x+nxcorr_y)/2
        
            try:
                maxpos = self.maxgauss(nxcorr,crop=self.crop,size=self.size)
            except (RuntimeError, TypeError):
                print("Error - curve_fit failed; try with individual fits")
                try:
                    maxpos1 = self.maxgauss(nxcorr_x,crop=self.crop,size=self.size)
                    maxpos2 = self.maxgauss(nxcorr_y,crop=self.crop,size=self.size)
                    maxpos = np.mean((maxpos1,maxpos2),axis=0)
                    print("Worked!")
                except (RuntimeError, TypeError):
                    maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
                    print("Error - curve_fit failed, i = %i, use maxpos" % i)
            x,y = [j-len(nxcorr)//2 for j in maxpos]
        
            if (abs(x)>self.maxshift) or (abs(y)>self.maxshift):
                try:
                    maxpos = self.maxgauss(nxcorr,crop=self.crop,size=self.size*2)
                except (RuntimeError, TypeError):
                    print("Error - curve_fit failed; try with individual fits")
                    try:
                        maxpos1 = self.maxgauss(nxcorr_x,crop=self.crop,size=self.size*2)
                        maxpos2 = self.maxgauss(nxcorr_y,crop=self.crop,size=self.size*2)
                        maxpos = np.mean((maxpos1,maxpos2),axis=0)
                        print("Worked!")
                    except (RuntimeError, TypeError):
                        print("Error - curve_fit failed, i = %i" % i)
                x,y = [j-len(nxcorr)//2 for j in maxpos]
                if (abs(x)>self.maxshift) or (abs(y)>self.maxshift):
                    print('Value for shift seems unreasonable')
            
            maxx.append(x)
            maxy.append(y)
            
            av_tip = np.mean(tip[i:i+2*average])
            av_tilt = np.mean(tilt[i:i+2*average])
            
            difpiston.append(av_tip*y+av_tilt*x)        
            
            
        if only_pos:
            return maxx, maxy
        if return_pos:
            return difpiston, maxx, maxy
        else:
            return difpiston
        
        
        
        
        
        
        
