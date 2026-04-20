#!/bin/bash

# Batch script for calculating the various OHT diagnostics using the python script
# calc_oht_diagnostics.py on the JASMIN batch system. See documenation of the python
# script for information about what physical diagnostics are calculated and what input
# variables are used.
#
# Note that some variables have been 'redacted', between << and >>. Also note that the
# scripts initially output individual members one year at a time due to the large input
# data volumes; additional tools (e.g., cdo mergetime) are used to concatenate and
# compress the outputs afterwards, but these commands are not included here.
#
# Scripts originally prepared by Jake R. Aylmer (January 2026)
# ------------------------------------------------------------------------------------- #

#SBATCH --partition=<<SET THIS>>
#SBATCH --account=<<SET THIS>>
#SBATCH --qos=<<SET THIS>>

#SBATCH --mem=15G
#SBATCH --job-name=CLE-OHT
#SBATCH --time=24:00:00
#SBATCH --output=log-%x-%j-%A_%a.stdout
#SBATCH --array=1-40

# Extra flags to python script:
flags="--ohtc-only"

# Ensemble member is job task ID:
members=(${SLURM_ARRAY_TASK_ID})

exps=("HIST2" "SSP370")

# Need to set these to the base location of the input and output data.
# For the input (dirIn), this is then followed by subdiretories of the
# experiment name (exps), ensemble-member number, then "OCN/yearly",
# then the year (I am just avoiding committing the JASMIN paths 
dirIn="<<SET THIS PATH>>"
dirOut="<<SET THIS PATH>>"

# Subdirectories of dirOut for output (also output netcdf file name prefixes):
dirOHC="ohc_col_tend"
dirOHTC="oht_con"
dirOHT="oht_lat"

# Python interpreter flags + script (assume we are executing from the same directory):
pyScript="-u ./calc_oht_diagnostics.py"

module load jaspy

for r in ${members[@]}
do
    for exp in ${exps[@]}
    do
        mkdir -p ${dirOut}/${dirOHC}/${exp}/${r}
        mkdir -p ${dirOut}/${dirOHTC}/${exp}/${r}
        mkdir -p ${dirOut}/${dirOHT}/${exp}/${r}

        if [[ "$exp" == "HIST2" ]]
        then
            years=$(seq 1950 2014)
        else
            years=$(seq 2015 2099)
        fi

        for y in ${years[@]}
        do
            # Workaround to deal with some duplicate files of sohefldo: make sure we get the
            # right one. This is rather specific to some duplicates in SSP370 for members:
            #
            #    r = 10, years 2038, 2046, 2048, 2052
            #    r = 17, year  2072
            #    r = 20, years 2060, 2072, 2086
            #    r = 21, year  2087
            #
            # The extra files start with "ORIG_" (r = 10, 17, 20) or "copy_" (r = 21). Can't
            # just use the "regular" prefixes to search for files as all members start with
            # a different prefix not obviously related to r.
            #
            hfdsIn=""
            for x in $(ls ${dirIn}/${exp}/${r}/OCN/yearly/${y}/*mon*sohefldo.nc)
            do
                if [[ "$(basename $x)" != "ORIG_"* ]] && [[ "$(basename $x)" != "copy_"* ]]
                then
                    hfdsIn=$(basename $x)
                    break
                fi
            done

            if [[ -f ${dirOut}/${dirOHT}/${exp}/${r}/${dirOHT}_mon_r${r}_y${y}.nc ]]
            then
                echo "${exp}/${r}/${dirOHT}_mon_r${r}_y${y}.nc exists; skipping"
            else
                python ${pyScript} --ohc-in   ${dirIn}/${exp}/${r}/OCN/yearly/${y}/*mon*opottemptend.nc      \
                                   --hfds-in  ${dirIn}/${exp}/${r}/OCN/yearly/${y}/${hfdsIn}                 \
                                   --ohc-out  ${dirOut}/${dirOHC}/${exp}/${r}/${dirOHC}_mon_r${r}_y${y}.nc   \
                                   --ohtc-out ${dirOut}/${dirOHTC}/${exp}/${r}/${dirOHTC}_mon_r${r}_y${y}.nc \
                                   --oht-out  ${dirOut}/${dirOHT}/${exp}/${r}/${dirOHT}_mon_r${r}_y${y}.nc   \
                                   ${flags}
            fi

        done  # years
    done  # exp
done  # member

exit 0
