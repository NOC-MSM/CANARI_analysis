# Sea ice area and extent diagnostics

This directory contains scripts for the calculation of sea ice area and extent diagnostics from the sea ice concentration output of the CANARI-LE. The main calculations are done in the python script `area_extent.py`, but this is called from the batch script `batch_area_extent.sh`. See documentation at the top of the python script for details.

Diagnostics available include total Arctic and total Southern Ocean sea ice area and sea ice extent, computed from either daily or monthly mean sea ice concentrations (model output variable `aice`), and regional versions of these based on the standard NSIDC region masks (which have been prepared on the native model grid separately).

