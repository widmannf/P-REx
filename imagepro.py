import numpy as np
from scipy import fftpack
import scipy.ndimage.filters
import scipy.optimize as opt


##################################################
## Class 2: Image processing functions
##################################################

class Imagepro:
    
    def __init__(self):
        pass
    
    def mask(self,indata,mask_val=0,mask_range=0.5,copy=True):
        """
        If the Input is a quadratic array, this function masks a circular area
        to simulate telescope data.
    
        If input is a 3d array, function masks every 2D array from it
        Mask value is 0 by default
        """
        if copy:
            data = np.copy(indata)
        else:
            data = indata
        
        if data.ndim == 3:
            if data.shape[1] != data.shape[2]:
                raise Exception('Array not quadratic')
            size = data.shape[1]
            length = len(data)
            x = np.linspace(-0.5,0.5,size)
            y = np.linspace(-0.5,0.5,size)
            mask, yy = np.meshgrid(x,y)
            r = np.sqrt(mask**2 + yy**2)
            mask[r>mask_range] = False
            mask[r<=mask_range] = True
            for i in range(length):
                data[i][mask==False] = mask_val
            return data
    
        elif data.ndim == 2:
            if data.shape[0] != data.shape[1]:
                raise Exception('Array not quadratic')
            size = data.shape[0]
            x = np.linspace(-0.5,0.5,size)
            y = np.linspace(-0.5,0.5,size)
            mask, yy = np.meshgrid(x,y)
            r = np.sqrt(mask**2 + yy**2)
            mask[r>mask_range] = False
            mask[r<=mask_range] = True
            data[mask==False] = mask_val
            return data
    
        else:
            raise Exception('wrong dimension of input array')
    
    
    

    def normalize(self,data,piston_free=True):
        """
        Normalization of each time step seperately, following Schoeck00
        Comment:
        The Weighing is not exactly the same as the std. deviation
        Real std. deviation would be:
            weight = np.average(np.power(data[i]-np.mean(data[i]),2))
    
        but with the used definition np.mean(np.power(dataset[i],2))
        equals one at every timestep, as demanded in Svchoeck00
    
        Both methods are the same, if data is piston free
        Can be enforced by piston_free
        
        Input & Output: Image array, 3D or 2D
        """
        if data.ndim == 3:
            one = data[0]
            av_data = np.zeros((len(data),one.shape[0],one.shape[1]))
            for i in range(len(data)):
                if piston_free:
                    weight = np.sqrt(np.average(np.power(data[i]-np.mean(data[i]),2)))
                    av_data[i] = (data[i]-np.mean(data[i]))/weight
                else:
                    weight = np.sqrt(np.average(np.power(data[i],2)))
                    av_data[i] = data[i]/weight
            return av_data
        elif data.ndim == 2:
            av_data = np.zeros_like(data)
            if piston_free:
                weight = np.sqrt(np.average(np.power(data-np.mean(data),2)))
                av_data = (data-np.mean(data))/weight
            else:
                weight = np.sqrt(np.average(np.power(data,2)))
                av_data = data/weight
            return av_data
        else:
            raise Exception('wrong dimension of input array')
        
    
    
    def _ndflip(self,a):
        """
        Inverts an n-dimensional array along each of its axes
        used for the cross-correlation as the cross-correlation 
        of functions f(t) and g(t) is equivalent to the convolution 
        of f*(−t) and g(t)
        """
        ind = (slice(None,None,-1),)*a.ndim
        return a[ind]
    
    
    
    def _procrustes(self,a,target,side='both',padval=0):
        """
        Forces an array to a target size by either padding it with a constant or
        truncating it

        Arguments:
            a: nput array of any type or shape
            target: Dimensions to pad/trim to, must be a list or tuple
        """
        try:
            if len(target) != a.ndim:
                raise TypeError('Target shape must have the same number of dimensions as the input')
        except TypeError:
            raise TypeError('Target must be array-like')

        try:
            #Get array in the right size to use
            b = np.ones(target,a.dtype)*padval
        except TypeError:
            raise TypeError('Pad value must be numeric')
        except ValueError:
            raise ValueError('Pad value must be scalar')

        aind = [slice(None,None)]*a.ndim
        bind = [slice(None,None)]*a.ndim

        # pad/trim comes after the array in each dimension
        if side == 'after':
            for dd in range(a.ndim):
                if a.shape[dd] > target[dd]:
                    aind[dd] = slice(None,target[dd])
                elif a.shape[dd] < target[dd]:
                    bind[dd] = slice(None,a.shape[dd])

        # pad/trim comes before the array in each dimension
        elif side == 'before':
            for dd in range(a.ndim):
                if a.shape[dd] > target[dd]:
                    aind[dd] = slice(a.shape[dd]-target[dd],None)
                elif a.shape[dd] < target[dd]:
                    bind[dd] = slice(target[dd]-a.shape[dd],None)

        # pad/trim both sides of the array in each dimension
        elif side == 'both':
            for dd in range(a.ndim):
                if a.shape[dd] > target[dd]:
                    diff = (a.shape[dd]-target[dd])/2.
                    aind[dd] = slice(int(np.floor(diff)),int(a.shape[dd]-np.ceil(diff)))
                elif a.shape[dd] < target[dd]:
                    diff = (target[dd]-a.shape[dd])/2.
                    bind[dd] = slice(int(np.floor(diff)),int(target[dd]-np.ceil(diff)))
        else:
            raise Exception('Invalid choice of pad type: %s' %side)
        b[bind] = a[aind]
        return b
    
    
    
    def xcorrelation(self,image,kernel,flip=True):
        """
        Cross Correlation of two images,
        Based on FFTs with padding to a size of 2N-1
        
        if flip uses the flip function to increase the speed
        """
        outdims = np.array([image.shape[dd]+kernel.shape[dd]-1 for dd in range(image.ndim)])
        if flip:
            af = fftpack.fftn(image,outdims)
            #for real data fftn(ndflip(t)) = conj(fftn(t)), but flipup is faster
            tf = fftpack.fftn(self._ndflip(kernel),outdims)
            # '*' in python: elementwise multiplikation 
            xcorr = np.real(fftpack.ifftn(tf*af))
        else:
            corr = fftpack.fftshift(fftpack.ifftn(np.multiply(fftpack.fftn(image,outdims),
                                                              np.conj(fftpack.fftn(kernel,outdims)))))
            xcorr = np.abs(corr)
        return xcorr
    
    
    def nxcorrelation(self,kernel,image,laplace=True,crop=False,cropval=2):
        """
        Normalized Cross Correlation of two images,
        Based on FFTs with padding to a size of 2N-1
        Searches the position of an image in a (original) kernel
    
        Normalization is done by dividing by a cross correlation
        of a constant matrix of the same size as the input.
        This accounts for the different overlapping at 
        different position.
        For information see e.g.:
        https://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation
        Schoeck98 (overlap factor in the paper is nearly identical to an AC of an
        array with ones)
    
        Laplace = if true applies a laplace fitler to the data
        Crop = if true croppes the outer pixel (how many given by crop value) to
        avoid boundary effects, when using laplace. (only necessary if resolution is bad, 
        bigger than Laplace filter)   
        """
        if laplace:
            if crop:
                image2 = scipy.ndimage.filters.laplace(image)[cropval:-cropval,cropval:-cropval]
                kernel2 = scipy.ndimage.filters.laplace(kernel)[cropval:-cropval,cropval:-cropval]
            else:
                image2 = scipy.ndimage.filters.laplace(image)
                kernel2 = scipy.ndimage.filters.laplace(kernel)
        else:
            image2 = image.astype(float)
            kernel2 = kernel.astype(float)
        xcorr = self.xcorrelation(kernel2,image2)
       
        ones1 = np.ones_like(image2)
        ones0 = np.ones_like(kernel2)
        onescorr = self.xcorrelation(ones0,ones1)
    
        nxcorr = xcorr/(np.std(image2)*np.std(kernel2)*onescorr)
        nxcorr = self._procrustes(nxcorr,kernel.shape,side='both')
    
        return nxcorr
    
    

    def cnxcorrelation(self,kernel,image,crop=False,laplace=True,cropval=2,mask_norm=True):
        """
        Same as Normalized Cross Correlation, but for circular data, padded with zeros
        around the available data
        
        If crop = True a quadratic area (as big as possible) is cutted out in order to 
        neglect effects from the circular form.
    
        Need to work on the normalization
        """
        if crop:
            if kernel.shape[0] != image.shape[0]:
                raise Exception('For cropping, images have to be the same size')
            if kernel.shape[1] != image.shape[1]:
                raise Exception('For cropping, images have to be the same size')
            if image.shape[0] != image.shape[1]:
                raise Exception('For cropping, images have to be quadratic')
            pixel = image.shape[0]
            boundary = int(math.ceil(pixel/2-pixel/(2*math.sqrt(2))))
            image = image[boundary:-boundary,boundary:-boundary]
            kernel = kernel[boundary:-boundary,boundary:-boundary]
            if laplace:
                image = scipy.ndimage.filters.laplace(image)
                kernel = scipy.ndimage.filters.laplace(kernel)
            xcorr = self.xcorrelation(kernel,image)
            ones = np.ones_like(image)
            onescorr = self.xcorrelation(ones,ones)
            nxcorr = xcorr/(np.std(image)*np.std(kernel)*onescorr)
            nxcorr = self._procrustes(nxcorr,kernel.shape,side='both')    
    
        else:
            if laplace:
                image2 = scipy.ndimage.filters.laplace(image)[cropval:-cropval,cropval:-cropval]
                image2 = self.mask(image2,mask_val=0)
                kernel2 = scipy.ndimage.filters.laplace(kernel)[cropval:-cropval,cropval:-cropval]
                kernel2 = self.mask(kernel2,mask_val=0)
            else:
                image2 = image
                kernel2 = kernel
            xcorr = self.xcorrelation(kernel2,image2)
            ones1 = np.ones_like(image2)
            ones0 = np.ones_like(kernel2)
            if mask_norm:
                ones1 = self.mask(ones1)
                ones0 = self.mask(ones0)
            onescorr = self.xcorrelation(ones0,ones1)
            nxcorr = xcorr/(np.std(image2)*np.std(kernel2)*onescorr)
            nxcorr = self._procrustes(nxcorr,kernel.shape,side='both')
        return nxcorr
    
    
    
    def _2Dgauss(self,xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        """
        function to fit & plot a 2D Gauss
        """
        (x, y) = xdata_tuple                                                        
        xo = float(xo)                                                              
        yo = float(yo)                                                              
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                          + c*((y-yo)**2)))                                   
        return g.ravel()
    
    
    
    def maxgauss(self,nxcorr,solid=False,returnall=False,crop=10,size=2,show=False):
        """
        Function to determine the maximum position of a fitted 2D Gaussian.
        Determines the maximum pixel value and crops a small image around this values
        Then fits a 2D gaussian and returns the postiton of the peak in pixels from the
        uncropped image.
    
        crop = in pixel, crop size around max value
        size = estimated width of gaussian, needs to be adapted, important for fit
        
        if solid:   returns maxpos if fit fails, else returns error (may be necessary to 
                    use error to try different fit
        if returnall: returns all parameter of fitted gauss, else only max position
        if show: prints out the image including the 2D Gaussian (for debugging & testing)
        """
        import matplotlib.pyplot as plt
        maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
        
        # Cut image to get a better result
        smallnxcorr = nxcorr[maxpos[0]-crop:maxpos[0]+crop,maxpos[1]-crop:maxpos[1]+crop]
        smalldimx = smallnxcorr.shape[0]
        smalldimy = smallnxcorr.shape[1]
    
        # Rescale if croped image moves out of bounds
        if smalldimx != smalldimy:
            while smalldimx != smalldimy:
                crop -= 1
                smallnxcorr = nxcorr[maxpos[0]-crop:maxpos[0]+crop,maxpos[1]-crop:maxpos[1]+crop]
                smalldimx = smallnxcorr.shape[0]
                smalldimy = smallnxcorr.shape[1]
    
    
        # Fit needs at least an array of 3x3
        # No idea how to handle that more elegant
        # so far gives the maximum pixel value if array gets smaller
        if crop < 3:
            print('Error: Peak to close to boundary, return maxpos values')
            return maxpos[0],maxpos[1]
            
        # Renew fit boundaries (necessary here, as size of cropped image may change)
        initial_guess = (0.7,smalldimx/2,smalldimy/2,size,size,0,0)
        x = np.linspace(0, smalldimx-1, smalldimx)
        y = np.linspace(0, smalldimy-1, smalldimy)
        x, y = np.meshgrid(x, y)
        smallnxcorr = smallnxcorr.reshape(smalldimx*smalldimy)
        if solid:
            try:
                popt, pcov = opt.curve_fit(self._2Dgauss, (x, y), smallnxcorr, p0=initial_guess)
            except (RuntimeError, TypeError):
                print('Error: Fit failed, return maxpos values')
            return maxpos[0],maxpos[1]
        else: 
            popt, pcov = opt.curve_fit(self._2Dgauss, (x, y), smallnxcorr, p0=initial_guess)
	
        popt[2] += (maxpos[0]-crop)
        popt[1] += (maxpos[1]-crop)
        
        if show:
            Z = np.arange(0,1,0.1)
            dimx = nxcorr.shape[0]
            dimy = nxcorr.shape[1]
            x = np.linspace(0, dimx-1, dimx)
            y = np.linspace(0, dimy-1, dimy)
            x, y = np.meshgrid(x, y)
            data_fitted = self._2Dgauss((x, y), *popt)
            fig, ax = plt.subplots(1, 1)    
            ax.hold(True)
            ax.imshow(nxcorr)
            ax.contour(x, y, data_fitted.reshape(dimx, dimy), colors='k',levels=Z)
            ax.axvline(popt[1],ls='--',color='k')
            ax.axhline(popt[2],ls='--',color='k')
            plt.show()

        if returnall:
            return popt
        else:
            return popt[2]-len(nxcorr)//2, popt[1]-len(nxcorr)//2

    




    def maxgauss_zoom(self, nxcorr, crop=3, size=0.5, zoom=1, order=1, show=False, returnall=False):
        """
        Same principle as maxgaussOnly difference: zooms in on cutted image to get more datapoints
        zoom gives the zooming factor and
        order the interpolation method
        
        Need to put the solid version in at some point, if I gonna use this regularly
        """
        import matplotlib.pyplot as plt
        maxpos = np.unravel_index(nxcorr.argmax(), nxcorr.shape)
            
        # Cut image to get a better result
        smallnxcorr = nxcorr[maxpos[0]-crop:maxpos[0]+crop,maxpos[1]-crop:maxpos[1]+crop]
        smalldimx = smallnxcorr.shape[0]
        smalldimy = smallnxcorr.shape[1]

        # Rescale if croped image moves out of bounds
        if smalldimx != smalldimy:
            while smalldimx != smalldimy:
                crop -= 1
                smallnxcorr = nxcorr[maxpos[0]-crop:maxpos[0]+crop,maxpos[1]-crop:maxpos[1]+crop]
                smalldimx = smallnxcorr.shape[0]
                smalldimy = smallnxcorr.shape[1]

        # zoom in
        smallnxcorr = scipy.ndimage.zoom(smallnxcorr, zoom, order=order)
        smalldimx *= zoom
        smalldimy *= zoom
    
        x = np.linspace(0, smalldimx-1, smalldimx)
        y = np.linspace(0, smalldimy-1, smalldimy)
        x, y = np.meshgrid(x, y)
        smallnxcorr = smallnxcorr.reshape(smalldimx*smalldimy)

        initial_guess = (0.5,smalldimx/2,smalldimy/2,size*zoom,size*zoom,0,0)

        popt, pcov = opt.curve_fit(self._2Dgauss, (x, y), smallnxcorr, p0=initial_guess)

        popt[1] = popt[1]/zoom+(maxpos[1]-crop)
        popt[2] = popt[2]/zoom+(maxpos[0]-crop)
    
        if show:
            Z = np.arange(0,1,0.1)
            dimx = nxcorr.shape[0]
            dimy = nxcorr.shape[1]
            x = np.linspace(0, dimx-1, dimx)
            y = np.linspace(0, dimy-1, dimy)
            x, y = np.meshgrid(x, y)
            data_fitted = self._2Dgauss((x, y), *popt)
            fig, ax = plt.subplots(1, 1)
            ax.hold(True)
            ax.imshow(nxcorr)
            ax.contour(x, y, data_fitted.reshape(dimx, dimy), colors='k',levels=Z)
            ax.axvline(popt[1],ls='--',color='k')
            ax.axhline(popt[2],ls='--',color='k')
            plt.show()
            
        if returnall:
            return popt
        else:
            return popt[2]-len(nxcorr)//2, popt[1]-len(nxcorr)//2

   
    
    
    def deconvolve(self,image, kernel):
        if image.shape != kernel.shape:
            kernel = self._procrustes(kernel,image.shape,side='both',padval=0)
        image_fft = fftpack.fftshift(fftpack.fftn(image))
        kernel_fft = fftpack.fftshift(fftpack.fftn(kernel))
        return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(image_fft/kernel_fft)))

    
