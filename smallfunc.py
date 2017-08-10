import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import glob
from scipy import fftpack
from os.path import expanduser

color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'



##################################################
## Part 0: different small functions for the use of prex
##################################################


# Matplotlib design
# include own Matplotlib style & a function to hide the unnecessary spines in the plots
def set_style(style):
    home = expanduser("~")
    style_path = home + "/Master/Mplstyle/"
    available_styles = [i[30:-9] for i in sorted(glob.glob(style_path+"*.mplstyle"))]
    if style in available_styles:
        style_path = style_path+style+'.mplstyle'
    elif style in plt.style.available:
        style_path = style
    else:
        raise ValueError("Style {} not found. Available styles: {}".format(style,", ".join(available_styles)))
    plt.rcParams.update(plt.rcParamsDefault)
    if style == 'output':
        plt.style.use("../Mplstyle/vdefault.mplstyle")
        plt.style.use("../Mplstyle/vpgf.mplstyle")
    else:
        plt.style.use(style_path)



def hide_spines(outwards=True,onlytop=False):
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        if onlytop:
            # Get all Axis instances related to the figure.
            for ax in figure.canvas.figure.get_axes():
                ax.spines['top'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                if outwards:
                    ax.spines['left'].set_position(('outward',10))
                    ax.spines['bottom'].set_position(('outward',10))
                    ax.spines['right'].set_position(('outward',10))
        else:
            # Get all Axis instances related to the figure.
            for ax in figure.canvas.figure.get_axes():
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                if outwards:
                    ax.spines['left'].set_position(('outward',10))
                    ax.spines['bottom'].set_position(('outward',10))
                
                
                

# Print process of loops
def print_status(number,total):
    number = number+1
    if number == total:
        print ("\rComplete: 100%")
    else:
        percentage = int((number/total)*100)
        print ("\rComplete: ", percentage, "%", end="")


# error mesurement
def rmse(prediction, target):
    """
    RMS error of two lists
    """
    prediction = np.array(prediction)
    target = np.array(target)
    return np.sqrt(((prediction - target) ** 2).mean())


def rms(values):
    """
    RMS of a list
    """
    zeros = np.zeros_like(values)
    return rmse(values,zeros)


def strehl(rms):
    strehl = math.exp(-(2*math.pi*rms)**2)
    return strehl



# (de)convolution of two images
def deconvo(data1,data2):
    deconvolution = np.real(deconvolve(data1, data2))
    maxi = np.max(deconvolution)  
    return deconvolution,maxi


def deconvolve(image, kernel):
    image_fft = fftpack.fftshift(fftpack.fftn(image))
    kernel_fft = fftpack.fftshift(fftpack.fftn(kernel))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(image_fft/kernel_fft)))


def convolve(image, kernel):
    image_fft = fftpack.fftshift(fftpack.fftn(image))
    kernel_fft = fftpack.fftshift(fftpack.fftn(kernel))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(image_fft*kernel)))


# determine properties of a vector
def length(v):
    return math.sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=math.acos(cosx) # in radians
    return rad*180/math.pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
    
    
