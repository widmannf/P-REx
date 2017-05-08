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




##################################################
## Class 5: Functions to use Prex directly on LBT-FLAO AO data
##################################################


from .pistonrec import *

def showFLAO(path='../FLAO_Data/'):
    filenames = sorted(glob.glob(path + '*.fits'))
    names = []
    for i in filenames:
        names.append(i[13:-5])
    return names

class FLAO(Prex):
    
    def __init__(self,path,name,size=0.5,crop=3,maxshift=4):
        self.path = path
        self.name = name
        self.size = size
        self.crop = crop
        self.maxshift = maxshift
        
        self.data = fits.open(self.path+self.name+'.fits')
        
        self.slopes      = self.data[1].data[:1376,:] 
        self.slopes_x    = self.data[2].data
        self.slopes_y    = self.data[3].data
        self.dm_modes    = self.data[4].data[:400,:] 
        self.dm_com      = self.data[5].data
        self.IM          = self.data[6].data[:1376,:]  
        self.CM          = self.data[7].data[:400,:1376] 
        self.c2m         = self.data[8].data
        self.m2c         = self.data[9].data
        self.indpup      = self.data[10].data
        self.dm_modes_ol = self.data[11].data
        self.CM2         = self.data[12].data[:400,:1376] 
        self.res_modes   = self.data[13].data
        
        self.lensletmask = np.zeros(80*80)
        self.lensletmask[self.indpup[2,:]]=1
        self.lensletmask = np.reshape(self.lensletmask,(80,80))
        self.lensletmask = self.lensletmask[7:37,7:37]
        
        #print('FLAO data initialized, data availabe as:' \
              #'\n   slopes, slopes_x, slopes_y, dm_modes,' \
              #'\n   dm_com, IM, CM, c2m, m2c, indpup, dm_modes_ol')
        
        
    def _2dimage(self,slopedata):
        """
        Takes list of slopes and puts it into a 2d image of size 30x30
        """
        screen = np.zeros(80*80)
        screen[self.indpup[2,:]]=slopedata
        screen2 = np.reshape(screen,(80,80))
        screen2 = screen2[7:37,7:37]
        return screen2
        
        
        
    def _oldata(self,redTT=True,twod=True):
        """
        Data includes OL DM data. This function puts this data into 
        2D slope data (for x and y slopes individually)
        Still nto sure how this OL data is created
        Output: tuple wih two arrays for x & y OL slopes
        """
        slopes_ol = np.dot(self.IM,self.dm_modes_ol)
        slopes_ol_x = slopes_ol[:1374:2,:]
        slopes_ol_y = slopes_ol[1:1374:2,:]
        
        if twod:
            time = slopes_ol_x.shape[1]
            xslopes_ol = np.zeros((time,30,30))
            yslopes_ol = np.zeros((time,30,30))
            for i in range(time):
                xslopes_ol[i] = self._2dimage(slopes_ol_x[:,i])
                yslopes_ol[i] = self._2dimage(slopes_ol_y[:,i])
                if redTT:
                    xslopes_ol[i] -= np.mean(xslopes_ol[i])
                    yslopes_ol[i] -= np.mean(yslopes_ol[i])
            return xslopes_ol[2:], yslopes_ol[2:]
        else:
            return slopes_ol_x[:,2:], slopes_ol_y[:,2:]
        
        
    def _cutimages(self,slopeimage,vx=1,vy=0,radius=6,distance=8,show_pos=False):
        """
        Uses a 2D image of slope data and cuts out 4 small circles
        (imaginary telescopes)
        the cut out circles are orientated with respect to the given wind vector, two in 
        its direction and two perpendicular to it. If no wind vector is given the 
        circles are in top/bottom and left/right. (x axis goes up)
        Distance from center and radius of the circles can be manipulated, but are by default at 
        the optimal values (circles as big as possible w/o empty datapoints) 
        
        show_pos: returns a image of the pupil with the position of the 4 circles
        """
        if (slopeimage.shape != (30, 30)):
            raise Exception('Error: unexpected shape of input data, should be (30,30)')
        v = np.array([vx,vy])
        norm = np.linalg.norm(v)
        nvx = vx*distance/norm
        nvy = vy*distance/norm
        nv = np.array([nvx,nvy])
        half = 15
        xshift = half + int(round(nvx))
        yshift = half + int(round(nvy))
        
        image1 = self.mask(slopeimage[xshift-radius:xshift+radius,yshift-radius:yshift+radius],
                           mask_val=0,mask_range=0.54)
        image2 = self.mask(slopeimage[-xshift-radius:-xshift+radius,-yshift-radius:-yshift+radius],
                           mask_val=0,mask_range=0.54)
        image3 = self.mask(slopeimage[-yshift-radius:-yshift+radius,xshift-radius:xshift+radius],
                           mask_val=0,mask_range=0.54)
        image4 = self.mask(slopeimage[yshift-radius:yshift+radius,-xshift-radius:-xshift+radius],
                           mask_val=0,mask_range=0.54)
        
        if show_pos:
            pupil = np.copy(slopeimage)
            pupil[pupil != 0 ] = 1
            pupil[pupil < 1 ] = np.nan
            pupil[pupil > 0 ] = 0
            pupil[xshift-radius:xshift+radius,yshift-radius:yshift+radius][image1 != 0] = 1
            pupil[-xshift-radius:-xshift+radius,-yshift-radius:-yshift+radius][image1 != 0] = 2
            pupil[-yshift-radius:-yshift+radius,xshift-radius:xshift+radius][image1 != 0] = 3
            pupil[yshift-radius:yshift+radius,-xshift-radius:-xshift+radius][image1 != 0] =  4
            return pupil, image1, image2, image3, image4
        
        else:
            return image1, image2, image3, image4
        
    def subimages(self,slopeimage,*args,**kwargs):
        """
        ...
        """
        dimx = len(slopeimage)
        subimage = np.zeros((dimx,4,12,12))
        for i in range(dimx):
            subimage[i] = self._cutimages(slopeimage[i],*args,**kwargs)
        return subimage
        
    def _poldata(self,twod=True,redTT=True,shift=2):
        """
        ...
        """
        pol = self.slopes[:,:-shift] + np.dot(self.IM,self.dm_modes)[:,shift:]
        # remove TT
        pol = pol[:-2,:]
        # split up x & y 
        pol_x = pol[::2,:]
        pol_y = pol[1::2,:]
        if twod:
            pol_x_2d = np.zeros((pol.shape[1],30,30))
            pol_y_2d = np.zeros((pol.shape[1],30,30))
            for i in range(pol.shape[1]):
                pol_x_2d[i] = self._2dimage(pol_x[:,i])
                pol_y_2d[i] = self._2dimage(pol_y[:,i])
                if redTT:
                    pol_x_2d[i] -= np.mean(pol_x_2d[i])
                    pol_y_2d[i] -= np.mean(pol_y_2d[i])

            return pol_x_2d, pol_y_2d
        else:
            return pol_x, pol_y
