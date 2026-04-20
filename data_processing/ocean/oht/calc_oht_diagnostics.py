"""Calculate diagnostics related to northward Ocean Heat Transport in the
CANARI Large Ensemble.

Inputs     opottemptend    Ocean potential temperature tendency, expressed as heat
                           content tendency in W/m2. This is a 3D field, i.e., a function
                           of time t, depth, and horizontal position y, x.
           sohefldo        Net downward surface heat flux into the ocean in W/m2.
                           This is a 2D field (t,y,x).
           areacello       Ocean grid cell areas (m2).
           subbasins       Ocean basin masks.

Outputs    ohc_col_tend    Depth-integrated ocean heat content tendency in W/m2 (t,y,x).
           oht_con         Ocean heat transport convergence in W/m2 (t,y,x).
           oht_lat         Ocean heat transport across latitudes in W (t,lat) computed
                           for the global ocean and in basins defined by the mask file
                           subbasins_corrected.

Pass option -h or --help to this script for options which specify the input and output
file names. Note the special flag --oht-only, which causes the script to assume the
required ohc_col_tend and oht_con fields have already been computed and skips to the
OHT calculation step.

"""

__author__ = "Jake R. Aylmer"

from argparse import ArgumentParser
import netCDF4 as nc
import numpy as np

# Latitudes at which to evaluate northward OHT:
lats_oht = np.arange(-80., 90.01, .5)

# Paths to files containing grid cell area (variable "areacello") and basin masks.
# Note the directory '../../../data' does not exist within the repository and is
# just a placeholder to save putting explicit paths on specific storage volumes in
# the code repository:
#
areafile  = "../../../data/ocean/areacello.nc"
basinfile = "../../../data/ocean/subbasins_corrected.nc"


def copy_nc_data(nc_src, nc_dst, exclude_dims=[], exclude_vars=[],
                 exclude_global_attrs=[]):
    """Copy all data and metadata from a source netCDF dataset instance to a destination
    one, except for any specified dimensions, variables, or global attributes.
    """

    # Global attributes:
    for name in nc_src.ncattrs():
        if name not in exclude_vars:
            nc_dst.setncattr(name, nc_src.getncattr(name))

    # Dimensions:
    for name, dimension in nc_src.dimensions.items():
        if name not in exclude_dims:
            if dimension.isunlimited():
                nc_dst.createDimension(name, None)
            else:
                nc_dst.createDimension(name, len(dimension))

    # Variables:
    for name, variable in nc_src.variables.items():
        if name not in exclude_vars:
            nc_dst.createVariable(name, variable.datatype, variable.dimensions)
            nc_dst.variables[name][:] = nc_src.variables[name][:]
            for attr in nc_src.variables[name].ncattrs():
                nc_dst.variables[name].setncattr(attr, nc_src.variables[name].getncattr(attr))


if __name__ == "__main__":

    # Command-line arguments: each is a path to one dataset (one netCDF file):
    prsr = ArgumentParser()
    prsr.add_argument("--ohc-in"  , type=str, required=True)
    prsr.add_argument("--hfds-in" , type=str, required=True)
    prsr.add_argument("--ohc-out" , type=str, required=True)
    prsr.add_argument("--ohtc-out", type=str, required=True)
    prsr.add_argument("--oht-out" , type=str, required=True)
    prsr.add_argument("--oht-only", action="store_true",
                      help="Skip OHCT/OHTC steps: presume they exist already")
    prsr.add_argument("--ohtc-only", action="store_true",
                      help="Skip OHCT step, go to OHTC and OHT calculations")
    cmd = prsr.parse_args()

    # Load cell area and basin masks:
    with nc.Dataset(areafile, "r") as ncdat:
        areacello = np.array(ncdat.variables["areacello"][:,:])

    msk_names = ["atlmsk", "pacmsk", "indmsk", "socmsk"]  # for output metadata

    with nc.Dataset(basinfile, "r") as ncdat:
        msk_atl = np.array(ncdat.variables["atlmsk"][:,:])
        msk_pac = np.array(ncdat.variables["pacmsk"][:,:])
        msk_ind = np.array(ncdat.variables["indmsk"][:,:])
        msk_soc = np.array(ncdat.variables["socmsk"][:,:])

    # Global mask: note this is *not* the same as just applying no mask, as the
    # basin masks also exclude lakes and certain other areas:
    msk_glo = msk_atl + msk_pac + msk_ind + msk_soc

    # Set non-ocean domain (global/any basin) in areacello to zero
    #
    # Note throughout: generally use 0 as the 'mask' value for calculations as we are
    # generally summing to get OHTs/areas (and so 0 excludes the grid cell) but in
    # output 2D fields (ohc_tend, ohtc) I use NaN instead
    #
    areacello = np.where(msk_glo < .5, 0., areacello)

    if cmd.oht_only:

        if cmd.ohtc_only:
            print("Computing OHT only (ignoring --ohtc-only)")
        else:
            print("Computing OHT only")

        # Skip to OHT calculation, but still need to get some data from
        # previous saved files:
        with nc.Dataset(cmd.ohtc_out, "r") as ncdat_in:
            ohtc = np.array(ncdat_in.variables["ohtc"][:,:,:])
            lat  = np.array(ncdat_in.variables["nav_lat"][:,:])

    else:

        if cmd.ohtc_only:
            print("Calculating OHTC and then OHT (skipping OHCT)")

            with nc.Dataset(cmd.ohc_out, "r") as ncdat_in:
                ohctend = np.array(ncdat_in.variables["ohctend"][:,:,:])

        else:
            print("Calculating OHCT, OHTC, then OHT (i.e., everything)")

            # Calculate column ocean heat content tendency. Input is opottemptend, which
            # claims to be potential temperature tendency dT/dt in K/s, but it is actually
            # the heat content tendency per unit area, i.e., rho*cp*(dT/dt)*dz in W.m-2,
            # where rho is density of sea water, cp is the specific heat capacity, and dz
            # is the layer thickness.
            #
            # So to get the total column heat content tendency, just need to sum along
            # the vertical dimension.
            #
            with nc.Dataset(cmd.ohc_in , "r") as ncdat_in , \
                 nc.Dataset(cmd.ohc_out, "w") as ncdat_out:

                # Get opottemptend data:
                opottemptend = np.array(ncdat_in.variables["opottemptend"][:,:,:,:])

                # Set missing to NaN (actual missing flag is 1.e20):
                opottemptend = np.where(opottemptend > 1.e19, np.nan, opottemptend)

                # Construct a land mask for final result. Note this is different to (the
                # inverse of) msk_glo above which includes masked-out lakes and some other
                # regions. Here it is not necessary to remove data for OHCT to be saved,
                # so just use the 'built in' land mask.
                #
                # (But later, for OHTC, save with the msk_glo applied so that the data is
                # consistent with the subsequently-calculated OHT over the various masks
                # msk_*)
                #
                lmask = np.where(np.isnan(opottemptend[0,0,:,:]), np.nan, 1.)

                # Calculate vertical integral sum (need nansum as some levels are missing
                # due to bathymetry) and apply land mask to result:
                ohctend = np.nansum(opottemptend, axis=1) * lmask[np.newaxis,:,:]

                # Copy everything needed from input to output netCDF file:
                copy_nc_data(ncdat_in, ncdat_out,
                             exclude_dims=["deptht"],
                             exclude_vars=["deptht", "deptht_bounds", "opottemptend"])

                # Now add the new variable:
                ncdat_out.createVariable("ohctend", ohctend.dtype, ("time_counter", "y", "x"))
                ncdat_out.variables["ohctend"][:,:,:] = ohctend[:,:,:]
                ncdat_out.variables["ohctend"].setncattr("cell_measures", "area: areacello")
                ncdat_out.variables["ohctend"].setncattr("cell_methods", "area: mean time_counter: mean")
                ncdat_out.variables["ohctend"].setncattr("coordinates", "nav_lat nav_lon")
                ncdat_out.variables["ohctend"].setncattr("long_name", "Ocean heat content tendency (depth integrated)")
                ncdat_out.variables["ohctend"].setncattr("units", "W m-2")

            print(f"Saved: {str(cmd.ohc_out)}")

        # [ End if cmd.ohtc_only ]

        # Compute OHT convergence from energetic considerations:
        #
        #     [column heat content tendency] = [Net downward surface heat flux] + [OHT convergence]
        #
        # The global mean of the resulting OHTC (*over msk_glo*) is then subtracted to
        # ensure the OHT calculated from it is zero at both poles. The non-zero global
        # mean is an error term that is also saved in the output NetCDF file so that the
        # 'true' OHTC in each grid cell calculated from the above budget can be recovered
        # (the typical value is small, less than 1 W/m2, at least in the historical period).
        #
        # The following comment is put in the output netCDF metadata:
        #
        ohtc_gm_comment = \
            ("Ocean heat transport convergence, variable \'ohtc\', is "
             + "calculated as a residual of the depth-integrated ocean heat "
             + "content tendency and net downward surface heat flux. In theory, "
             + "the global mean of this should be zero; to ensure this is so, "
             + "the global mean of the calculation is removed in \'ohtc\'. This "
             + "variable, \'ohtc_gm\', contains the values of those original "
             + "global means as a function of time so that the original residual "
             + "can be recovered if required. All basin masks have been applied "
             + "to this data and \'global mean\' refers to this domain, not the "
             + "original model output ocean domain.")

        # For OHC tendency, set out-of-global-domain values to 0 for further
        # calculations (it is currently NaN) and apply msk_glo:
        ohctend = np.where(np.isnan(ohctend), 0., ohctend)
        ohctend = np.where(msk_glo[np.newaxis,:,:] < .5, 0., ohctend)

        with nc.Dataset(cmd.hfds_in, "r") as ncdat_in, \
             nc.Dataset(cmd.ohtc_out, "w") as ncdat_out:

            # Load net downward surface heat flux data ('sohefldo') and
            # calculate OHTC, also building in global mask msk_glo:
            ohtc = np.where(msk_glo[np.newaxis,:,:] < .5, 0.,
                            ohctend[:,:,:] - ncdat_in.variables["sohefldo"][:,:,:])

            # Subtract global mean of OHTC from itself. This ensures OHT goes to zero at
            # both poles. Note msk_glo is already implicitly included in areacello as
            # zeros. Save value as scalar variable in output so it can be added back later
            # if needed for any reason:
            ohtc_gm = np.sum(ohtc*areacello[np.newaxis,:,:], axis=(1,2)) / np.sum(areacello)
            ohtc -= ohtc_gm[:,np.newaxis,np.newaxis]

            # For some reason, here the global integral of ohtc is not exactly 0. It is tiny
            # and definitely negligible (~1 W cf. PW scale)
            # (precision/noise? Everything is float64)

            # Copy everything needed from input to output netCDF file:
            copy_nc_data(ncdat_in, ncdat_out, exclude_vars=["sohefldo"])

            ncdat_out.setncattr("external_variables",
                                ncdat_out.getncattr("external_variables") + "; "
                                + "; ".join(msk_names))

            # Now add the new variables:
            ncdat_out.createVariable("ohtc_gm", ohtc_gm.dtype, ("time_counter",))
            ncdat_out.variables["ohtc_gm"][:] = ohtc_gm[:]
            ncdat_out.variables["ohtc_gm"].setncattr("cell_methods", "time_counter: mean")
            ncdat_out.variables["ohtc_gm"].setncattr("comment", ohtc_gm_comment)
            ncdat_out.variables["ohtc_gm"].setncattr("long_name", "Global mean of OHTC")
            ncdat_out.variables["ohtc_gm"].setncattr("units", "W m-2")

            ncdat_out.createVariable("ohtc", ohtc.dtype, ("time_counter", "y", "x"))
            ncdat_out.variables["ohtc"][:,:,:] = np.where(msk_glo[np.newaxis,:,:] < .5, np.nan, ohtc)
            ncdat_out.variables["ohtc"].setncattr("cell_measures", "area: areacello")
            ncdat_out.variables["ohtc"].setncattr("cell_methods", "area: mean time_counter: mean")
            ncdat_out.variables["ohtc"].setncattr("coordinates", "nav_lat nav_lon")
            ncdat_out.variables["ohtc"].setncattr("long_name",
                "Ocean heat transport convergence (depth integrated, global mean subtracted)")
            ncdat_out.variables["ohtc"].setncattr("units", "W m-2")

            # Extract latitudes for calculation of OHT below:
            lat = np.array(ncdat_in.variables["nav_lat"][:,:])

        print(f"Saved: {str(cmd.ohtc_out)}")

    # [ End if cmd.oht_only ]

    #            # ============= #
    #            # NORTHWARD OHT #
    #            # ============= #

    # Pre-calculate OHTC multiplied by grid cell area:
    ohtc_areacello = ohtc * areacello[np.newaxis,:,:]

    # Calculate total northward OHT across latitudes by integrating (summing) from
    # latitude and everywhere northward of it, with relevant basin mask. Southern
    # Ocean is tricky because of non-zero northern-most boundary condition and the
    # inclusion of the southern coast of Australia (Great Bight) in its mask, so
    # I just subtract the other components from the global to get it afterwards:
    #
    oht_glo = np.zeros( (len(ohtc_areacello), len(lats_oht)) )
    oht_atl = np.zeros_like(oht_glo)
    oht_pac = np.zeros_like(oht_glo)
    oht_ind = np.zeros_like(oht_glo)

    for j in range(len(lats_oht)):
        # Mask for everywhere northward of current latitude (lats_oht[j]):
        lat_msk = np.where(lat > lats_oht[j], 1., 0.)

        for data, bas_msk in zip([oht_glo, oht_atl, oht_pac, oht_ind],
                                 [msk_glo, msk_atl, msk_pac, msk_ind]):
            data[:,j] = np.sum(ohtc_areacello * lat_msk[np.newaxis,:,:]
                                              * bas_msk[np.newaxis,:,:],
                               axis=(1,2))

    # =============================    Note     ============================= #
    # The following post-processing steps on all but the global OHT are
    # specific to the masks as defined in the subbasins_corrected.nc file
    # (i.e., where exactly the boundaries are placed).
    # ======================================================================= #

    # Need to mask out (set to NaN) the OHTs at latitudes for which the basin
    # masks of the basin OHTs are undefined.
    #
    # Identify the lowest (lo) and highest (hi) latitudes of the basins where
    # required (for Atlantic and Pacific hi is the north pole, and lo for
    # Southern Ocean is the land boundary of Antarctica, so they are not needed):
    #
    atl_lat_lo = np.nanmin(np.where(msk_atl < .5, np.nan, lat))  # \
    pac_lat_lo = np.nanmin(np.where(msk_pac < .5, np.nan, lat))  #  > should be the same
    ind_lat_lo = np.nanmin(np.where(msk_ind < .5, np.nan, lat))  # /
    ind_lat_hi = np.nanmax(np.where(msk_ind < .5, np.nan, lat))
    soc_lat_hi = np.nanmax(np.where(msk_soc < .5, np.nan, lat))

    # Find the indices of lats_oht that correspond to each of those:
    jatl_lo = np.argmax(lats_oht >= atl_lat_lo)
    jpac_lo = np.argmax(lats_oht >= pac_lat_lo)
    jind_lo = np.argmax(lats_oht >= ind_lat_lo)
    jind_hi = np.argmin(lats_oht <= ind_lat_hi)
    jsoc_hi = np.argmin(lats_oht <= soc_lat_hi)

    # These indices on axis=1 of oht_* are now used to mask out the values outside the
    # basins, e.g., for Indian basin we want it to look like this:
    #
    # [ ..., NaN, NaN, Val, ..., Val, NaN, NaN, ... ]
    #                   ^         ^
    #                   |         |
    #                jind_lo   jind_hi
    #
    # where the Vals are values computed above and all those that we want to
    # set to NaN are currently zero or some constant value.

    oht_atl[:,:jatl_lo]   = np.nan
    oht_pac[:,:jpac_lo]   = np.nan
    oht_ind[:,:jind_lo]   = np.nan
    oht_ind[:,jind_hi+1:] = np.nan

    # Calculate Southern Ocean heat transport. Subtract other basins from the global,
    # replacing the NaNs just assigned with zeros (could not calculate this more masking
    # above because, e.g., Atlantic heat trasports below lats_oht[:jatl_lo] == lats_oht[jatl_lo]):
    #
    oht_soc =   np.where(np.isnan(oht_glo), 0., oht_glo) \
              - np.where(np.isnan(oht_atl), 0., oht_atl) \
              - np.where(np.isnan(oht_pac), 0., oht_pac) \
              - np.where(np.isnan(oht_ind), 0., oht_ind)

    # Mask the SO heat transport:
    oht_soc[:,jsoc_hi+1:] = np.nan

    # ======================================================================= #

    # Comment to describe each OHT variable in the netCDF metadata, with format
    # placeholders {} for the basin and then mask description:
    #
    oht_nc_comment = \
        ("{} northward meridional ocean heat transport (OHT) calculated by "
         + "integrating OHT convergence (OHTC) northward of native grid cell-"
         + "center latitudes{}, where OHTC is computed on the native grid "
         + "from the residual of net flux into the ocean column (sohefldo) "
         + "and depth-integrated ocean heat content tendency (opottemptend, "
         + "summed vertically), and the global mean of the OHTC has been "
         + "subtracted before computation.")

    with nc.Dataset(cmd.ohtc_out, "r") as ncdat_src, nc.Dataset(cmd.oht_out, "w") as ncdat_out:

        copy_nc_data(ncdat_src, ncdat_out,
                     exclude_dims=["x", "y", "nvertex"],
                     exclude_vars=["x", "y", "bounds_nav_lat", "bounds_nav_lon",
                                   "ohtc", "ohtc_gm", "nav_lon", "nav_lat"])

        ncdat_out.setncattr("external_variables",
                            ncdat_out.getncattr("external_variables") + "; "
                            + "; ".join(msk_names))

        ncdat_out.createDimension("lat", None)
        # (unlimited because latitudes here are arbitrary)

        ncdat_out.createVariable("lat", lats_oht.dtype, ("lat",))
        ncdat_out.variables["lat"].setncattr("long_name", "latitude")
        ncdat_out.variables["lat"].setncattr("standard_name", "latitude")
        ncdat_out.variables["lat"].setncattr("units", "degrees_north")
        ncdat_out.variables["lat"][:] = lats_oht

        for x, n1, n2, data in zip(
                ["", "_atl", "_pac", "_ind", "_soc"],
                ["Net (all basins)", "Atlantic", "Pacific", "Indian Ocean", "Southern Ocean"],
                ["", " and over the Atlantic Ocean mask", " and over the Pacific Ocean mask",
                     " and over the Indian Ocean mask"  , " and over the Southern Ocean mask"],
                [oht_glo, oht_atl, oht_pac, oht_ind, oht_soc]):
            ncdat_out.createVariable(f"oht{x}", data.dtype, ("time_counter", "lat"))
            ncdat_out.variables[f"oht{x}"][:,:] = data[:,:]

            ncdat_out.variables[f"oht{x}"].setncattr("cell_methods", "lat: point time_counter: mean")
            ncdat_out.variables[f"oht{x}"].setncattr("comment", oht_nc_comment.format(n1, n2))
            ncdat_out.variables[f"oht{x}"].setncattr("long_name", f"{n1} northward ocean heat transport")
            ncdat_out.variables[f"oht{x}"].setncattr("standard_name", "northward_ocean_heat_transport")
            ncdat_out.variables[f"oht{x}"].setncattr("units", "W")

    print(f"Saved: {str(cmd.oht_out)}")

