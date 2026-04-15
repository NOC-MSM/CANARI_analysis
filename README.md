# CANARI_diagnostics

A public repository for developing and sharing code for diagnosing data from observations and models in the CANARI project (https:www.canari.ac.uk).

This CANARI sub-project code repository, "CANARI_diagnostics" is open to suggested changes (philosophical, structural, stylistic, ...) and was initialised simply to aid collaboration.

Initially, it's likely to be mainly NOC (noc.ac.uk) staff working on the CANARI project who might use it, but others in the CANARI consortium and beyond are also welcome too.

To get started, clone this Git repository, e.g. from the Linux command line : 

`git clone git@github.com:NOC-MSM/CANARI_diagnostics.git`

or, if that doesn't work (as may be the case on JASMIN), try the https form:

`git clone https://github.com/NOC-MSM/CANARI_diagnostics.git`


**Contributors - please add code in either the Python or Fortran directories, accordingly.  Or add a new directory for other types of code.  Also - add at least a one-line description of what your code does, in this README file, below, sorted in alphabetical order.**

e.g. 
FORTRAN/ice_extent.F90 : calculates the area (km^2) based on the sea-ice concentration greater than a threshold value (0.15 default), including non-uniform grid metric terms for the area elements.
FORTRAN/ice_extent.h90 : header file for ice_extent.F90
FORTRAN/grid_define_convert.F90 : general utility module for defining and converting grids, used by other modules (ice_extent.F90, ...)

Python/ice_extent.ipynb : calculates the area (km^2) based on the sea-ice concentration greater than a threshold value (0.15 default), including non-uniform grid metric terms for the area elements.


## Some useful and relevant sources of information:

### NOC's Research Computing Community channel on Microsoft Teams

[Research Computing Community](https://teams.microsoft.com/l/team/19%3avUGcIO9U_W-wdpVEw1XdVwsALb4J_ZRoBTqXOjtfJvU1%40thread.tacv2/conversations?groupId=4a9592c5-b621-4e26-a86a-8afaacf82d7c&tenantId=b8fab1d4-ed23-4180-9900-2d2a7880e9ba)

### General data analysis

https://docs.xarray.dev/en/stable/

### Model grid-aware

https://xgcm.readthedocs.io/en/latest/

### For model-observational verification or model-model validation

https://british-oceanographic-data-centre.github.io/COAsT/

https://github.com/NOC-MSM/NOC_Near_Present_Day
