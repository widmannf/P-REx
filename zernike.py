###########
## This file is part of the Python Module P-Rex, a module for test for a 
## piston reconstruction experiement for optical interferometer (see Pott et al 2016)
## This files contains several functions for the use of zernike polynomials
##
## Copyright (c) 2017, Felix Widmann
##
## This program is free software; you can redistribute it and/or  modify it
## under the terms of the GNU General Public License  as  published  by the
## Free Software Foundation; either version 2 of the License,  or  (at your
## option) any later version.
##
##
## Some of this functions are on basis of the libtim package from Tim van Werkhoven
## see: https://github.com/tvwerkhoven/libtim-py
## as well as code from Ravi S. Jonnal, see:
## https://github.com/rjonnal/zernike/blob/master/zernike/__init__.py
###########



import numpy as np
from scipy.misc import factorial as fac
import math

from .smallfunc import *

##################################################
## Class 1: functions for the use of Zernike modes
##################################################

class Zernike:
    
    def __init__(self):
        pass
    
    
    def _noll2zern(self, j):
        """
        Calculate the n and m Zernike ciefficient from a given Noll number
        Input: j: Noll number
        Output: n,m: Zernike coeficients
        """
        if (j < 1):
            raise ValueError("Noll indices start at 1.")
        n = 0
        j1 = j-1
        while (j1 > n):
            n += 1
            j1 -= n
        m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
        return (m,n)
    
    
    def _radzernike(self, m, n, r):
        """
        Calculate the radial part of the Zernike polynomials
        """
        if (np.mod(n-m, 2) == 1):
            return np.zeros_like(r)
        wf = np.zeros_like(r)
        for k in range((n-m)//2+1):
            wf += r**(n-2.0*k) * (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
        return wf


    
    def zernike(self, m, n, u, r, norm=True): 
        """
        Calculate the radial part of the Zernike polynomials
        """
        nc = 1.0
        if (norm):
            nc = (2*(n+1)/(1+(m==0)))**0.5
        if m > 0: 
            zk = nc*self._radzernike(m, n, r) * np.cos(m * u)
        elif m < 0: 
            zk = nc*self._radzernike(-m, n, r) * np.sin(-m * u)
        else:
            zk = nc*self._radzernike(0, n, r)
        return zk


    
    def modes2WF(self, modes, pixel=100, withoutTT=False, mask_size=1):
        """
        Calclulate the wavefront from a set of zernike modes with Noll numbers
        
        Input: array of zernike modes in the Noll order starting with piston
    
        pixel: Diameter of the final image in pixel
        withoutTT: set to True to set the first three coeffiecents to zero
        
        Output: 2D array, image of wavefront
        """
        z_modes = np.copy(modes)
        # Regular, cartesian grid
        x = np.linspace(-1,1,pixel)
        y = np.linspace(-1,1,pixel)
        xx, yy = np.meshgrid(x,y)
        xx2 = np.copy(xx)
        yy2 = np.copy(yy)
        # Mask values outside unit circle
        xx2[xx**2 + yy**2 > mask_size] = np.nan
        yy2[xx**2 + yy**2 > mask_size] = np.nan
        # Calculate polar coordinate arrays
        u = np.arctan2(yy2,xx2)
        r = np.sqrt(xx2**2 + yy2**2)    
        # Calculate polynoms for this grid
        Z = np.zeros_like(r)
        if withoutTT:
            z_modes[0:3] = 0
        for idx, item in enumerate(z_modes):
            m,n = self._noll2zern(idx+1)    
            Z += item * self.zernike(m,n,u,r)
        return Z



    #########################
    ##### The following functions are used to
    ##### reconstruct phase front from Slopes
    ##### Typical command for this:
    #####   # create reconstructor matrix:
    #####   s2z = zernrexonstructor_givenmask(numerSlopes,numberZernModes,pupilImage)
    #####   # Build wavefront
    #####   WF = slopes2WF(s2z,xSlopes,ySlopes,pupilImage)
    #########################
    
    
    def _choose(self,a,b):
        """Binomial coefficient, implemented using
        this module's factorial function.
        See [here](http://www.encyclopediaofmath.org/index.php/Newton_binomial) for detail.
        """
        assert(a>=b)
        return math.factorial(a)/(math.factorial(b)*math.factorial(a-b))

    def _zernikeparams(self,n,m,kind=0):
        """
        Based on code from: Ravi S. Jonnal
        https://github.com/rjonnal/zernike/blob/master/zernike/__init__.py
        
        Return parameters sufficient for specifying a Zernike term
        of desired order and azimuthal frequency.
    
        Given an order (or degree) n and azimuthal frequency f, and x-
        and y- rectangular (Cartesian) coordinates, produce parameters
        necessary for constructing the appropriate Zernike
        representation.
        An individual polynomial has the format:
    
         Z_n^m = \sqrt{c} \Sigma^j\Sigma^k [a_{jk}X^jY^k] 
    
        This function returns a tuple (c,cdict). c is the square
        of the normalizing coefficient \sqrt{c}, and cdict contains
        key-value pairs ((j,k),a), mapping the X and Y
        exponents (j and k, respectively) onto polynomial term
        coefficients (a). The resulting structure can be used to
        compute the wavefront height or slope for arbitrary pupil
        coordinates, or to generate string representations of the
        polynomials.

        Args:
        n (int): The Zernike order or degree.
        m (int): The azimuthal frequency.
        kind (str): 0, 1 or 2, for height (0), partial x
          derivative (1) or partial y derivative (2)
          respectively.
        Returns:
        params (tuple): (c,cdict), with c being the normalizing
                coefficient c and cdict being the map of exponent pairs
                onto inner coefficients.
        """
        absm = np.abs(m)
        kindIndex = kind

        # Check if n and m are valid zernike indices
        # check that n and m are both even or both odd
        if (float(n-absm))%2.0:
            raise Exception('Error: parity of n and m are different; n = %d, m = %d'%(n,m))
        # check that n is non-negative:
        if n<0:
            raise Exception('Error: n must be non-negative')
        # |m| must be less than or equal to n.
        if abs(m)>n:
            raise Exception('Error: |m| must be less than or equal to n')
        
        
        # These are the squares of the outer coefficients. It's useful
        # to keep them this way for _convertToString, since we'd
        # prefer to print the $\sqrt{}$ rather than a truncated irrational
        # number.
        if m==0:
            outerCoef = n+1
        else:
            outerCoef = 2*(n+1)
        
        srange = range((n-absm)//2+1)
        cdict = {}
        for s in srange:
            jrange = range(((n-absm)//2)-s+1)
            for j in jrange:
            # Subtract 1 from absm to determine range,
                # only when m<0.
                if m<0:
                    krange = range((absm-1)//2+1)
                else:
                    krange = range(absm//2+1)
                for k in krange:
                    # If m==0, k must also be 0;
                    # see eqn. 13c, 19c, and 20c, each of which
                    # only sum over s and j, not k.
                    if m==0:
                        assert(k==0)
                    # For m==0 cases, n/2 is used in coef denominator. Make
                    # sure that n is even, or else n/2 is not well-defined
                    # because n is an integer.
                    if m==0:
                        assert n%2==0
                
                
                    # The coefficient for each term in this
                    # polynomial has the format: $$\frac{t1n}{t1d1
                    # t1d2 t1d3} t2 t3$$. These six terms are
                    # computed here.
                    t1n = ((-1)**(s+k))*math.factorial(n-s)
                    t1d1 = math.factorial(s)
                    t1d2 = math.factorial((n + absm)/2-s)
                    t1d3 = math.factorial((n - absm)/2-s)
                    t1 = t1n/(t1d1*t1d2*t1d3)
                        
                    t2 = self._choose((n - absm)/2 - s, j)
                    t3 = self._choose(absm, 2*k + (m<0))
                    if kind == 0:
                        # The (implied) coefficient of the $X^a Y^b$
                        # term at the end of eqns. 13a-c.
                        c = 1 
                        tXexp = n - 2*(s+j+k) - (m<0)
                        tYexp = 2*(j+k) + (m<0)
                    elif kind == 1:
                        # The coefficient of the $X^a Y^b$ term at
                        # the end of eqns. 19a-c.
                        c = (n - 2*(s+j+k) - (m<0)) 

                        # Could cacluate explicitly:
                        # $tXexp = X^{(n - 2*(s+j+k)- 1 - (m<0))}$
                        # However, piggy-backing on previous
                        # calculation of c speeds things up.
                        tXexp = c - 1
                        tYexp = 2*(j+k) + (m<0)
                    elif kind == 2:
                        # The coefficient of the $X^a Y^b$ term at
                        # the end of eqns. 20a-c.
                        c = 2*(j+k) + (m<0)
                        tXexp = n - 2*(s+j+k) - (m<0)
                        tYexp = c - 1
                    else:
                        raise Exception('Error: invalid kind %i; should be 0, 1 or 2.'%kind)
 

                    ct123 = c*t1*t2*t3
                    # The key for the polynomial dictionary is the pair of X,Y
                    # coefficients.
                    termKey = (tXexp,tYexp)

                    # Leave this term out of th e dictionary if its coefficient
                    # is 0.
                    if ct123:
                        # If we already have this term, add to its coefficient.
                        if termKey in cdict:
                            cdict[termKey] = cdict[termKey] + ct123
                        # If not, add it to the dictionary.
                        else:
                            cdict[termKey] = ct123
        # Remove zeros to speed up computations later.
        cdict = {key: value for key, value in cdict.items() if value}
        return (outerCoef,cdict)




    def getSurface(self,n,m,X,Y,kind=0,mask=None):
        """
        Return a phase map specified by a Zernike order and azimuthal frequency
        Given an order (or degree) n and azimuthal frequency f, and x- and y-
        rectangular (Cartesian) coordinates, produce a phase map of either height,
        partial x derivative, or partial y derivative.
        Zernike terms are only defined when n and m have the same parity (both odd
        or both even).
        Args:
        n (int): The Zernike order or degree.
        m (int): The azimuthal frequency.
        X (float): A scalar, vector, or matrix of X coordinates in unit pupil.
        Y (float): A scalar, vector, or matrix of Y coordinates in unit pupil.
        kind (str): 1,2 or 3 for height, partial x derivative (slope)
          or partial y derivative, respectively.
        Returns:
        float: height, dx, or dy; returned structure same size as X and Y.
        """

        # Check that shapes of X and Y are equal (not necessarily square).
        if not np.all(X.shape==Y.shape):
            raise Exception('Error: X and Y must have the same shape')
    
        if mask is None:
            mask = np.ones(X.shape)
        params = self._zernikeparams(n,m,kind)
        normalizer = np.sqrt(params[0])
        matrix_out = np.zeros(X.shape)
    
        for item in params[1].items():
            matrix_out = matrix_out + item[1] * X**(item[0][0]) * Y**(item[0][1])
        matrix_out = matrix_out * normalizer
        matrix_out = matrix_out * mask
        return matrix_out



    def zernrexonstructor(self,numSlopes,numZernike,nlensletsdiam,returnMask=True):
        """
        Build a matrix to reconstruct Zernike modes from a list of slopes
        Matrix will have a size of #slopes x #zernikemodes
        
        One may have to be carefull with the definition of the mask and the way the
        slopes are used
        """
        jvec = range(numZernike)
        nmvec = np.array([self._noll2zern(x+1) for x in jvec])
    
        # Build Mask
        lensletedges = np.linspace(-1.0,1.0,nlensletsdiam+1)
        lensletcenters = (lensletedges[1:]+lensletedges[:-1])/2.0
        lensletXX,lensletYY = np.meshgrid(lensletcenters,lensletcenters)
        lensletD = np.sqrt(lensletXX**2+lensletYY**2)
        lensletmask = np.zeros(lensletD.shape)
        lensletmask[np.where(lensletD<1.0)]=1
    
        # Get Matrix
        A = np.zeros([numSlopes*2,numZernike])
        for j in jvec:
            m, n = self._noll2zern(j+1)
            dzdx = self.getSurface(n,m,lensletXX,lensletYY,1)
            dzdy = self.getSurface(n,m,lensletXX,lensletYY,2)
            dzdx = dzdx[np.where(lensletmask)]
            dzdy = dzdy[np.where(lensletmask)]
            A[:numSlopes,j] = dzdx
            A[numSlopes:,j] = dzdy
    
        B = np.linalg.pinv(A)
        if returnMask:
            return B, lensletmask
        else:
            return B
        
    def zernrexonstructor_givenmask(self,numSlopes,numZernike,lensletmask):
        """
        Build a matrix to reconstruct Zernike modes from a list of slopes
        Matrix will have a size of #slopes x #zernikemodes
        
        Input has to include a mask of the slope distribution
        """
        jvec = range(numZernike)
        nmvec = np.array([self._noll2zern(x+1) for x in jvec])
        lensletedges = np.linspace(-1.0,1.0,len(lensletmask)+1)
        lensletcenters = (lensletedges[1:]+lensletedges[:-1])/2.0
        lensletXX,lensletYY = np.meshgrid(lensletcenters,lensletcenters)
    
        # Get Matrix
        A = np.zeros([numSlopes*2,numZernike])
        for j in jvec:
            m, n = self._noll2zern(j+1)
            dzdx = self.getSurface(n,m,lensletXX,lensletYY,1)
            dzdy = self.getSurface(n,m,lensletXX,lensletYY,2)
            dzdx = dzdx[np.where(lensletmask)]
            dzdy = dzdy[np.where(lensletmask)]
            A[:numSlopes,j] = dzdx
            A[numSlopes:,j] = dzdy
    
        B = np.linalg.pinv(A)
        return B


    def slopes2WF(self,B,xslopes,yslopes,mask,pixel=200,*args,**kwargs):
        """
        Reconstruct a wavefroint directly from the measured slopes
        using zernike polynomials
        
        input:
        B: matrix from _zernreconstruction
        x & y slopes
        used mask
        """
        if xslopes.ndim == 2:
            lxslopes = xslopes[np.where(mask)]
            lyslopes = yslopes[np.where(mask)]
            slopes = np.hstack((lxslopes,lyslopes))
        else:
            slopes = np.hstack((xslopes,yslopes))
    
        rVec = np.dot(B,slopes)
        rwavefront = self.modes2WF(rVec,pixel=pixel,*args,**kwargs)
        rwavefront -= np.nanmean(rwavefront)
        return rwavefront
