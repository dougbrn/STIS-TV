# STIS-TV (Space Telescope Imaging Spectrograph Temperature Variability)
Modelling the dark rate temperature sensitivity of Hubble's Space Telescope Imaging Spectrograph (STIS)

Since the failure of the primary side-1 set of electronics in Jun 2001, STIS has operated on it's redundant set of side-2 electronics. While built to mirror the side-1 set, side-2 lacks the ability to measure the temperature on the CCD. Without this, we are forced to push a constant current to the Thermoelectric Cooler (TEC) which introduces temperature variability to our CCD. Furthermore, without the ability to measure our CCD temperature we are forced to use the CCD Housing temperature as a proxy. With these challenges in mind, this repository serves as a documentation point and storage space for the code and findings of my investigation into the sensitivity of the STIS CCD to changes in the housing temperature.

The Instrument Science Report (ISR) that this project culminated in has been published here: http://www.stsci.edu/hst/stis/documents/isrs/2018_05.pdf
