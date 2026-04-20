#!/bin/bash

# Batch script for calculating the various sea ice area and extent diagnostics using the
# python script area_extent.py on the JASMIN batch system. See documenation of the python
# script for information about what physical diagnostics are calculated and what input
# variables are used.
#
# Note that some variables have been 'redacted', between << and >>.
#
# Scripts originally prepared by Jake R. Aylmer (March 2024)
# ------------------------------------------------------------------------------------- #

#SBATCH --partition=<<SET THIS>>
#SBATCH --account=<<SET THIS>>
#SBATCH --qos=<<SET THIS>>

#SBATCH --mem=60G
#SBATCH --time=02:00:00

#SBATCH --job-name=CLE-SIE
#SBATCH --output=%x-%j.stdout

# Ensemble members (one output file per member):
members=$(seq 1 40)

# Input data is daily ('day') or monthly ('mon'):
freq="day"

# If processing daily data, do it in batches of dy years starting from year ys and ending
# in year ye. The way this is currently coded below, dy must be a factor of (ye - ys + 1),
# i.e., for 1950-2014, which is 65 years, choose dy = 1, 5, or 13. Note this will change
# memory requirements above, which depends essentially on dy. For processing monthly data
# this doesn't matter (i.e., just set dy = ye - ys + 1):
#
ys=1950  # start year
ye=2014  # end year
dy=5     # number of years to process in one go

# Experiment: must match next sub-directory under dataDir (below)
# then ./CICE/yearly/YYYY/...
exp="HIST2"

# Python options and script which actually does calculations (assume we run this
# bash script in the same working directory):
pyScript="-u -W ignore ./area_extent.py"

# Need to set these to the base location of the input and output data.
# For the input (dataDir), this is then followed by subdiretories of the
# experiment name (exp), ensemble-member number, then "CICE/yearly", then the
# year (I am just avoiding committing the JASMIN paths to the repository):
#
dataDir="<<SET THIS PATH>>"
outDir="<<SET THIS PATH>>"

mkdir -p ${outDir}/${exp}

for r in ${members[@]}
do
    for y in $(seq ${ys} ${dy} ${ye})
    do
        if [[ -f ${outDir}/${exp}/_siextent_diags_${freq}_${y}-$((y+dy-1))_r${r}.nc ]]
        then
            # Do not overwrite existing (may have interrupted/timed-out previous batch job):
            echo "${outDir}/${exp}/_siextent_diags_${freq}_${y}-$((y+dy-1))_r${r}.nc exists; skipping"
        else
            # Get array of input files for this set of dy years:
            infiles=()
            for yy in $(seq ${y} $((y+dy-1)))
            do
                infiles+=(${dataDir}/${exp}/${r}/CICE/yearly/${yy}/*_${freq}_aice.nc)
            done

            # See docs of pyScript for usage/flag meanings:
            python ${pyScript} -v -i ${infiles[@]}                                     \
                -d sia_n sie_n sia_s sie_s sia_reg_n sie_reg_n                         \
                -o ${outDir}/${exp}/_siextent_diags_${freq}_${y}-$((y+dy-1))_r${r}.nc
        fi
    done

    # Combine all years into one file for this one member
    #
    # Note: CDO removes a scalar coordinate variable called "siconc_threshold" which is
    # created during the python script (not sure how to get it to *not* do that, but
    # it's not essential for that variable to be there -- it's unlikely that the
    # threshold for extent would ever be anything other than 0.15):
    cdo --no_history -w mergetime                                           \
        ${outDir}/${exp}/_siextent_diags_${freq}_????-????_r${r}.nc         \
        ${outDir}/${exp}/siextent_diagnostics_${freq}_${ys}-${ye}_r${r}.nc

    # Remove intermediate files:
    rm ${outDir}/${exp}/_siextent_diags_${freq}_????-????_r${r}.nc

    echo "Saved: ${outDir}/${exp}/siextent_diagnostics_${freq}_${ys}-${ye}_r${r}.nc"
done

exit 0
