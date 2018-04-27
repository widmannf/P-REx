
# P-REx

This repository contains the Python module developed for the piston reconstruction experiment (P-REx).
P-REx reconstructs the piston drift due to the atmosphere above a single aperture in order to improve the sensitivity of optical interferometer. This module was developed as part of my master thesis.

Further details can be found in the paper by Pott et al (2016, http://adsabs.harvard.edu/abs/2016SPIE.9907E..3EP) and Widmann et al (2017, http://adsabs.harvard.edu/abs/2018MNRAS.475.1224W).

More information on the code in the Python files.
A complimentary repository contains my master thesis and an example notebook for the use of this module. 


## Usage:
Pull directory to pythonpath, then:
import prex
p = prex.Prex()

## Short Introduction:

The main function is the piston reconstruction based on the Tip-Tilt method:

diffpiston = p.prexTT(datacube,average)
Datacube is a list of the measured atmospheric data from the wavefront sensor in the order:
x-slopes, y-slopes, tip, tilt

**The slopes have to be a list/array of 2D arrays, in open loop or pseudo open loop!**

The function then does the following:
- average over slopes
- cross correlation of the x- and y-slopes in order to determine the wind vector
- calculates the differential fit from the measured wind vecor and the tip/tilt
