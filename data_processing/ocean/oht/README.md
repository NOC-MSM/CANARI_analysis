# Ocean heat transport (OHT) diagnostics

This directory contains scripts for the calculation of northward ocean heat transport (OHT) from outputs of the CANARI-LE. The main calculations are done in the python script `calc_oht_diagnostics.py`, but this is called from the batch script `batch_oht_diagnostics.sh`. See documentation at the top of the python script for details.

The primary diagnostic is net (integrated over depth and longitude) northward OHT as a function of time and latitude, computed for the global ocean as well as the Atlantic, Pacific, Indian, and Southern Ocean basins (for which there is a pre-defined mask file). It is calculated from the net surface heat flux into the ocean column and the column-integrated heat content tendency, using energetic considerations. As such, 'by-products' of the latter and OHT convergence (2D maps) are also saved. The outputs are monthly resolution.

