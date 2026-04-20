"""Calculate sea ice extent and area for pan Arctic, Southern Ocean, and/or specified
regions. It is designed to work on the CANARI-LE data and as such some variable names
and metadata are hard-coded.

Originall produced for the CANARI-LE Sprint analysis, March 2024.


Example usage
-------------
$ python siextent.py -v -i siconc_data_*.nc -d sie_n sie_s -o sie_out.nc

reads sea ice concentration raw data from siconc_data_*.nc, calculates pan-Arctic (sie_n)
and Southern hemisphere (sie_s) sea ice extent and saves to sie_out.nc. Remove the -v
(--verbose) flag to suppress console output.

Multiple inputs, to be concatenated with respect to time, are allowed:
-i file_1.nc file_2.nc [...] or implicitly via wildcards.

Multiple output files (-o flag) can be specified to separate output data as long as there
are the same number of -d flags provided, each of which can contain any number of
diagnostics including duplicates. For example:

$ python siextent.py -v -i siconc_data_*.nc -d sie_n sia_n
                                            -d sie_s sia_s
                                            -d sie_n sia_n sie_s sia_s
                                            -o nhemi.nc shemi.nc all.nc

saves extent and area diagnostics to a separate file for each hemisphere and then a third
file all.nc with all diagnostics saved together.

"""

__author__ = "Jake R. Aylmer"

from argparse import ArgumentParser
from datetime import datetime as dt

import netCDF4 as nc
import numpy as np
from numpy import newaxis as npna  # short alias for convenience


# Note: directory '../../data' does not actually exist within repository, it is
# just a placeholder for the purposes of putting the script into the repository
# without having to hard-code in paths on specific storage volumes:
areacello_file = "../../data/ocean/areacello.nc"
regions_file   = "../../data/seaice/ssmi/auxilliary/sea_ice_regions.nc"

# List of lists of diagnostics [see parse_args() documentation)]:
default_diagnostic_lists = [["sia_n", "sie_n"], ["sie_reg_n"]]

sia_kw = {"standard_name": "sea_ice_area"  , "units": "1e6 km2"}
sie_kw = {"coordinates"  : "siconc_threshold",
          "standard_name": "sea_ice_extent", "units": "1e6 km2"}

nc_var_attrs = {
    "sia_n"     : {"long_name": "Pan Arctic sea ice area"      , **sia_kw},
    "sia_s"     : {"long_name": "Southern Ocean sea ice area"  , **sia_kw},
    "sie_n"     : {"long_name": "Pan Arctic sea ice extent"    , **sie_kw},
    "sie_s"     : {"long_name": "Southern Ocean sea ice extent", **sie_kw},
    "sia_reg_n" : {"long_name": "Sea ice area (regional)"      , **sia_kw},
    "sie_reg_n" : {"long_name": "Sea ice extent (regional)"    , **sie_kw}
}

# Scalar coordinate variable used for sea ice extent outputs:
nc_siconc_threshold_attrs = {"standard_name": "sea_ice_area_fraction", "units": "1"}

# For regional sea ice extent/area:
nc_regions_external_variable_name = "sea_ice_regions"
nc_region_attrs = {}

# Inherit the following global netCDF attributes from the (first) input file for
# output files (if these don't exist, they are set to "N/A", but most of these
# will exist in all CANARI-LE model output files):
inherit_global_attrs = ["activity_id", "branch_method", "branch_time_in_child",
    "branch_time_in_parent", "comment", "experiment", "forcing_index", "further_info_url",
    "grid", "grid_label", "initialization_index", "institution", "nominal_resolution",
    "owner", "parent_experiment_id", "parent_mip_era", "parent_source_id",
    "parent_variant_label", "physics_index", "realization_index", "realm", "source",
    "source_index", "source_type", "title", "variant_id"]

# Crop rows of siconc data before calculating sea ice extent/area (speeds it up;
# one for each for the northern NH and southern SH hemispheres):
j_crop_n = 800  # ~30N; uses siconc[:, j_crop_n:, :] for NH
j_crop_s = 550  # ~30S; uses siconc[:, :j_crop_s, :] for SH

# For regional sea ice extent/area (NH only), a separate, hard-coded value
# (of j_crop_n_reg = 800) is used because of the way the region masks have
# been computed.
#
# =========================================================================== #


def vprint(x, verbose=False, **kwargs):
    if verbose:
        print(x, **kwargs)


def parse_args():
    """"Define and get command-line arguments using argparse."""

    prsr = ArgumentParser("Calculate sea ice extent/area")

    # Input files. This is just a straightforward, space-separated list of
    # inputs from the command line after -i or --infiles flag:
    prsr.add_argument("-i", "--infiles", type=str, nargs="*", help="File paths")

    # Diagnostic list(s) to calculate in groups corresponding to output files (other
    # option -o/--outfiles below). This is a list of lists, specified on the command line
    # as one list per -d flag, e.g.,
    #
    # -d <diag_1_in_outfile_1> <diag_2_in_outfile_1> ...
    # -d <diag_1_in_outfile_2> <diag_2_in_outfile_2> ...
    # ...
    #
    prsr.add_argument("-d", "--diagnosticlists", type=str, nargs="+", action="append",
                      choices=list(nc_var_attrs.keys()),
                      help="List of diagnostic(s) to save")

    # Output netCDF files for each group of diagnostics specified by -d/--diagnostics:
    prsr.add_argument("-o", "--outfiles", type=str, nargs="*",
                      help="Output netCDF file(s) (paths) corresponding to diagnostics"
                           + " list(s) (-d/--diagnosticlist)")

    # Usually, do not need to change this:
    prsr.add_argument("-t", "--siconcthreshold", type=float, default=0.15,
                      help="Sea ice concentration threshold for extent calculations")

    prsr.add_argument("-v", "--verbose", action="store_true")

    args = prsr.parse_args()

    return args


def load_siconc_data(input_files,
                     get_global_nc_attrs = inherit_global_attrs,
                     verbose             = True):
    """Load and concatenate sea ice concentration, and time and other metadata, from
    input files. Grid coordinates are not loaded (get these from the cell area file).


    Note
    ----
    I normally just use netCDF4 and have a separate function that loads and concatenates
    specified fields; this is adapted here. Note this assumes that input file names, when
    sorted, are in the correct time order (the module has an MFDataset class for multiple
    inputs but it only works with netCDF3 and netCDF4-classic formats). It is probably
    possible to make this part more efficient with xarray or some other package.
    Alternatively, concatenating the raw data first can be used (e.g., CDO mergetime).


    Parameters
    ----------
    input_files: iterable of str, length >= 1
        Input raw netCDF file(s). These are sorted before loading and concatenating in
        order of input (see note above)


    Optional parameters
    -------------------
    get_global_nc_attrs : iterable of str
        NetCDF global attributes to read and return from the first input file, stored in
        a dictionary (giving the return value 'nc_global_attrs').

    verbose : bool, default = True
        Whether to print loading progress to the console.


    Returns
    -------
    nc_global_attrs : {global_attribute : <value>}
        Dictionary of global attributes loaded from the first input file (i.e., assume
        they are the same for all inputs)

    nc_time_attrs : {"bounds": "time_bnds", "units": <str>, "calendar": <str>}
        Attributes needed to set the time and time_bnds variable in the output.

    time : array (nt,) of datetime
        Time coordinates.

    time_bnds : array (nt, 2) of cftime
        Time coordinate bounds (i.e., averaging intervals).

    siconc : array (nt, ny, nx) of float or double
        Raw sea ice concentration data.

    """

    input_files = sorted(list(input_files))

    nc_global_attrs = {x: "" for x in get_global_nc_attrs}

    # Load the first dataset explicitly to get dimensions, coordinates,
    # and initialise arrays:
    with nc.Dataset(input_files[0], "r") as ncdat_0:

        vprint(f"Loading: {input_files[0]}", verbose)

        for attr in get_global_nc_attrs:
            try:
                nc_global_attrs[attr] = getattr(ncdat_0, attr)
            except AttributeError:
                nc_global_attrs[attr] = "N/A"
                vprint(f"Warning: no global attribute \"{attr}"
                       + "\" found; setting to \"N/A\"",verbose)

        time_units = getattr(ncdat_0.variables["time"], "units")
        time_cal = getattr(ncdat_0.variables["time"],"calendar")

        # Load time as dates (in case units or calendar are different per file):
        date = nc.num2date(np.array(ncdat_0.variables["time"][:]),
                           units=time_units, calendar=time_cal)

        date_bnds = nc.num2date(np.array(ncdat_0.variables["time_bounds"][:,:]),
                                units=time_units, calendar=time_cal)

        siconc = np.array(ncdat_0.variables["aice"][:,:,:])

    for j in range(1, len(input_files)):
        vprint(f"Loading and appending: {input_files[j]}", verbose)

        with nc.Dataset(input_files[j], "r") as ncdat_j:
            time_units_j = getattr(ncdat_j.variables["time"], "units")
            time_cal_j   = getattr(ncdat_j.variables["time"], "calendar")

            date = np.concatenate((date,
                                   nc.num2date(np.array(ncdat_j.variables["time"][:]),
                                               units=time_units_j, calendar=time_cal_j)))

            date_bnds = np.concatenate((date_bnds,
                nc.num2date(np.array(ncdat_j.variables["time_bounds"][:,:]),
                            units=time_units_j, calendar=time_cal_j)), axis=0)

            siconc = np.concatenate((siconc,
                                     np.array(ncdat_j.variables["aice"][:,:,:])), axis=0)

    # Use first input file for time units and calendar:
    nc_time_attrs = {"bounds": "time_bnds", "calendar": time_cal, "units": time_units}

    # Convert dates back to time:
    time = nc.date2num(date, units=time_units, calendar=time_cal)

    time_bnds = nc.date2num(date_bnds, units=time_units, calendar=time_cal)

    return nc_global_attrs, nc_time_attrs, time, time_bnds, siconc


def load_grid_data(expected_shape, verbose=True):
    """Load grid data: longitude, latitude, and cell areas (all from the cell area file).


    Parameters
    ----------
    expected_shape : tuple (ny, nx) of int
        Array dimensions expected used to infer whether the cell area data loaded has
        halo cells or not. Should be the spatial part of the shape of sea ice
        concentration data loaded. Note that a ValueError is raised if the (final)
        shape of areacello does not match this input.

    verbose: bool, default = True
        Print progress to the console.


    Returns
    -------
    lon : array, shape (ny, nx)
        Longitude coordinates (degrees_east presumed).

    lat : array, shape (ny, nx)
        Latitude coordinates (degrees_north presumed).

    areacello : array, shape (ny, nx)
        Grid cell areas (m2 presumed).

    """

    vprint(f"Loading cell area from {areacello_file}", verbose)

    with nc.Dataset(areacello_file, "r") as ncdat:
        lon       = np.array(ncdat.variables["nav_lon"][:,:])
        lat       = np.array(ncdat.variables["nav_lat"][:,:])
        areacello = np.array(ncdat.variables["areacello"][:,:])

    ny, nx = expected_shape

    if np.shape(areacello) == (ny+2, nx+2):
        # Remove halo cells from areacello (data from CICE does not have those):
        lon = lon[1:-1, 1:-1]
        lat = lat[1:-1, 1:-1]
        areacello = areacello[1:-1, 1:-1]

    if np.shape(areacello) != expected_shape:
        raise ValueError(  f"Shape of areacello data {np.shape(areacello)} does not "
                         + f"match that of siconc {expected_shape}")

    return lon, lat, areacello


def load_regions_data(verbose=True):
    """Load array that defines regions by an integer value."""

    vprint(f"Loading regions mask from {regions_file}", verbose)

    with nc.Dataset(regions_file, "r") as ncdat:
        reg_data = np.array(ncdat.variables["sea_ice_region"])

    # The saved regions array comes from the NSIDC definitions which are on an Arctic
    # stereographic grid (i.e., not global). I interpolated it onto a cropped version of
    # the model grid; specifically, keeping rows from j=800 only, which corresponds to
    # about 30N to 90N (i.e., safely covers northern hemisphere sea ice while removing
    # more than half of the domain).
    #
    # This makes sense for the observation data (keeps data volume/interpolation time
    # low) and also makes sense to use here to speed up computation time of regional
    # extent/area.
    #
    # So, return this value for use in other routines (i.e., to crop raw sea ice
    # concentration data to the same domain):
    j_crop = 800

    # Integer array corresponding to region flag (see external file for definitions;
    # it goes from 0-18):
    regions_use = np.arange(19).astype(np.int32)
    nreg = len(regions_use)

    # Compute mask array for each region ID:
    regions_mask = np.zeros((nreg, *np.shape(reg_data)))

    for r in range(nreg):
        regions_mask[r,:,:] = np.where(abs(reg_data - regions_use[r]) < .5, 1., np.nan)

    return j_crop, regions_use, regions_mask


def prepare_siconc(lon, lat, siconc, miss_val=1.e19, verbose=True):
    """Prepare raw sea ice concentration for processing.

    This currently sets missing values to NaN, removes invalid values, and masks out key
    areas such as lakes which contain ice in the sea ice concentration output field. The
    area masking is only done here for the Arctic, but other masking regions can be
    defined and added. Note this makes a small but noticeable difference in winter months
    pan-Arctic extent. Masking areas just check grid cell center coordinates, not cell
    bounds.

    The mask region definitions come from another set of Python code that was used in
    some published work, archived here:

        https://zenodo.org/doi/10.5281/zenodo.5494523


    Parameters
    ----------
    lon : array (ny, nx) of float or double
        Longitude coordinates (degrees_east).

    lat : array (ny, nx) of float or double
        Latitude coordinates (degrees_north).

    siconc : array (nt, ny, nx) of float or double
        Sea ice concentration raw data (units are assumed to be fraction -- which is the
        case for the HadGEM3-GC31-MM data -- not percentage).


    Optional parameters
    -------------------
    miss_val : float, default = 1.e19
        This is not the true missing value flag, but anything greater than this is
        assumed to be missing.

    verbose : bool, default = True
        Whether to print progress to the console.


    Returns
    -------
    siconc_prep : array (nt, ny, nx)
        Prepared sea ice concentration data for extent/area calculations to be applied.

    """

    vprint("Preparing siconc data", verbose)

    siconc_prep = np.where(siconc > miss_val, np.nan, siconc)
    siconc_prep = np.maximum(0., siconc_prep)
    siconc_prep = np.minimum(1., siconc_prep)

    # Mask lakes/other regions that might have sea ice but are not wanted by setting them
    # to NaN. Need longitude in 0-360 range for this:
    lon_check = lon % 360.

    # --------------------------------------------------------------------------------- #
    # Docs/data below extracted from ice_edge_latitude utilities.regions.py module at:  #
    # https://zenodo.org/doi/10.5281/zenodo.5494523                                     #
    # --------------------------------------------------------------------------------- #
    #
    # Regions are specified as follows:
    #
    #     example_region = {"Sub-region 1 name" : pos_1,
    #                       "Sub-region 2 name" : pos_2,
    #                       ...
    #                      }
    #
    # where
    #
    #     pos_x = [lon_min, lon_max, lat_min, lat_max]
    #
    # specifies a lon-lat boundary of sub-region x. In this way, arbitrary geometries can
    # be specified or collections of related regions define as one.
    #
    # The keys are not used for anything so just act as comments to know what each pos_x
    # refers is for. However, they must be unique (within each region).
    #
    # Longitudes are in degrees east (i.e., always positive and between 0 and 360).
    # --------------------------------------------------------------------------------- #
    mask_regions = {"lakes"     : {"L. Balkhash": [70, 80, 40, 50],
                                   "L. Aike"    : [57, 62, 43, 47],
                                   "Caspian Sea": [45, 55, 35, 48]},
                    "baltic_sea": {"North Baltic Sea" : [15, 30, 60, 66],
                                   "South Baltic Sea" : [10, 30, 53, 60]},
                    "black_sea" : {"Black Sea" : [26, 41, 40, 47]}
    }
    # --------------------------------------------------------------------------------- #

    for reg in mask_regions.keys():
        for sub_reg in mask_regions[reg].keys():
            sub_reg_mask = np.where(  (lon_check >= mask_regions[reg][sub_reg][0])
                                    & (lon_check <= mask_regions[reg][sub_reg][1])
                                    & (lat       >= mask_regions[reg][sub_reg][2])
                                    & (lat       <= mask_regions[reg][sub_reg][3]),
                                    np.nan, 1.)

            siconc_prep *= sub_reg_mask[npna,:,:]

    return siconc_prep


def sea_ice_area(siconc, areacello, units_factor=1.e-12):
    """Calculate sea ice area (total area of sea ice, equal to the sum of all grid cell
    areas, areacello, each multiplied by their sea ice concentrations, siconc. Mask
    siconc before passing.
    """
    return units_factor * np.nansum( siconc*areacello[npna,:,:], axis=(1,2) )


def sea_ice_extent(siconc, areacello, threshold=0.15,
                   units_factor=1.0E-12):
    """Calculate sea ice extent (total area of all grid cell areas, areacello, where sea
    ice concentration, siconc, is at least equal to specified threshold. Mask siconc
    before passing.
    """
    extent = np.where(siconc >= threshold, 1., 0.)
    return units_factor * np.nansum( extent * areacello[npna,:,:], axis=(1,2) )


def set_nc_attrs(nc_something, attrs):
    """Set attributes on a nc file or variable, nc_something, using attributes specified
    in an input dictionary, attrs.
    """
    for attr in sorted(list(attrs.keys())):
        setattr(nc_something, attr, attrs[attr])


def is_extent(x):
    """Check that given diagnostic name x is extent."""
    return "ie" in x


def is_regional(x):
    """Check that given diagnostic name x is regional."""
    return "reg" in x


def main():

    cmd = parse_args()  # command-line arguments
    verbose = cmd.verbose

    dt_start = dt.now()
    vprint(f"Start: {dt_start.strftime('%H:%M, %a %d %b %Y')}", verbose)

    # If no diagnostics specified on the command line, set default here:
    if cmd.diagnosticlists is None:
        cmd.diagnosticlists = default_diagnostic_lists
    
    if len(cmd.outfiles) != len(cmd.diagnosticlists):
        raise ValueError(  f"Number of output file names must match number of diagnostic "
                         + f"lists; got {len(cmd.outfiles)} file names and "
                         + f"{len(cmd.diagnosticlists)} diagnostic lists")

    if cmd.infiles is None:
        raise ValueError("Must provide at least one input file -i <input_data.nc>")

    # Boolean flag for each diagnostic list corresponding to whether they have any sea
    # ice extent diagnostics:
    has_extent = [any([is_extent(x) for x in dlist]) for dlist in cmd.diagnosticlists]

    # Similarly for regional diagnostics:
    has_regional = [any([is_regional(x) for x in dlist]) for dlist in cmd.diagnosticlists]

    # Unique, ordered list of diagnostics specified:
    all_diagnostics = sorted(list(set([x for dlist in cmd.diagnosticlists for x in dlist])))

    in_files = sorted(list(cmd.infiles))

    # Load sea ice concentration and time data:
    nc_global_attrs, nc_time_attrs, time, time_bnds, siconc = \
        load_siconc_data(cmd.infiles, verbose=verbose)

    nt, ny, nx = np.shape(siconc)

    # Load grid data (must come after sea ice):
    lon, lat, areacello = load_grid_data((ny, nx), verbose)

    # Load region data (if required):
    if any(has_regional):
        j_crop_n_reg, regions_use, regions_mask = load_regions_data(verbose)
        nreg = len(regions_use)

    # Prepare siconc data:
    siconc = prepare_siconc(lon, lat, siconc)

    # Masks for northern/southern hemispheres:
    nhemi = np.where(lat >= 0., 1., 0.)
    shemi = np.where(lat <  0., 1., 0.)

    # Save each diagnostic array into this dictionary (for use when saving to netCDF):
    save_data = {}

    if "sia_n" in all_diagnostics:

        vprint("Calculating " + nc_var_attrs["sia_n"]["long_name"], verbose)

        save_data["sia_n"] = sea_ice_area(siconc[:,j_crop_n:,:]*nhemi[npna,j_crop_n:,:],
                                          areacello[j_crop_n:,:])

    if "sia_s" in all_diagnostics:
        
        vprint("Calculating " + nc_var_attrs["sia_s"]["long_name"], verbose)

        save_data["sia_s"] = sea_ice_area(siconc[:,:j_crop_s,:]*shemi[npna,:j_crop_s,:],
                                          areacello[:j_crop_s,:])

    if "sie_n" in all_diagnostics:
        
        vprint("Calculating " + nc_var_attrs["sie_n"]["long_name"], verbose)

        save_data["sie_n"] = sea_ice_extent(siconc[:,j_crop_n:,:]*nhemi[npna,j_crop_n:,:],
                                            areacello[j_crop_n:,:],
                                            threshold=cmd.siconcthreshold)

    if "sie_s" in all_diagnostics:
        
        vprint("Calculating " + nc_var_attrs["sie_s"]["long_name"], verbose)

        save_data["sie_s"] = sea_ice_extent(siconc[:,:j_crop_s,:]*shemi[npna,:j_crop_s,:],
                                            areacello[:j_crop_s,:],
                                            threshold=cmd.siconcthreshold)

    if any(has_regional):
        if "sia_reg_n" in all_diagnostics:
            save_data["sia_reg_n"] = np.zeros((nt, nreg))

        if "sie_reg_n" in all_diagnostics:
            save_data["sie_reg_n"] = np.zeros((nt, nreg))

        if "sia_reg_n" in all_diagnostics:
            for r in range(nreg):
                vprint("Calculating " + nc_var_attrs["sia_reg_n"]["long_name"]
                       + f": region {r+1:02}/{nreg:02}" + "\r", verbose, end="")

                save_data["sia_reg_n"][:,r] = sea_ice_area(
                    siconc[:,j_crop_n_reg:,:]*regions_mask[[r],:,:],
                    areacello[j_crop_n_reg:,:])

            vprint("", verbose)

        if "sie_reg_n" in all_diagnostics:
            for r in range(nreg):
                vprint("Calculating " + nc_var_attrs["sie_reg_n"]["long_name"]
                       + f": region {r+1:02}/{nreg:02}" + "\r", verbose, end="")

                save_data["sie_reg_n"][:,r] = sea_ice_extent(
                    siconc[:,j_crop_n_reg:,:]*regions_mask[[r],:,:],
                    areacello[j_crop_n_reg:,:], threshold=cmd.siconcthreshold)

            vprint("", verbose)

    # --------------------------------------------------------- #
    # Save data
    # ========================================================= #
    for j in range(len(cmd.outfiles)):

        nc_global_attrs_j = nc_global_attrs.copy()

        with nc.Dataset(cmd.outfiles[j], "w") as ncdat:

            ncdat.createDimension("time", None)
            ncdat.createDimension("bnd" , 2)

            ncdat.createVariable("time", time.dtype, ("time",))
            ncdat.createVariable(nc_time_attrs["bounds"], time_bnds.dtype, ("time", "bnd"))

            set_nc_attrs(ncdat.variables["time"], nc_time_attrs)

            ncdat.variables["time"][:] = time
            ncdat.variables["time_bnds"][:,:] = time_bnds

            # Add a region dimension and coordinate variable if needed (i.e., if the set
            # of diagnostics for this output file includes any regional diagnostics):
            if has_regional[j]:
                ncdat.createDimension("region", None)
                ncdat.createVariable("region", np.int32, ("region",))

                ncdat.variables["region"][:] = regions_use

                set_nc_attrs(ncdat.variables["region"], nc_region_attrs)

                nc_global_attrs_j["external_variables"] = nc_regions_external_variable_name

            # Set scalar coordinate variable for threshold if this nc file has any
            # extent-related diagnostics:
            if has_extent[j]:
                ncdat.createVariable("siconc_threshold", np.float32)

                ncdat.variables["siconc_threshold"][:] = cmd.siconcthreshold

                set_nc_attrs(ncdat.variables["siconc_threshold"], nc_siconc_threshold_attrs)

            for x in sorted(cmd.diagnosticlists[j]):
                if is_regional(x):
                    ncdat.createVariable(x, save_data[x].dtype, ("time", "region"))
                    ncdat.variables[x][:,:] = save_data[x]
                else:
                    ncdat.createVariable(x, save_data[x].dtype, ("time",))
                    ncdat.variables[x][:] = save_data[x]

                set_nc_attrs(ncdat.variables[x],nc_var_attrs[x])

            set_nc_attrs(ncdat, nc_global_attrs_j)

        print(f"Saved: {str(cmd.outfiles[j])}")

    dt_end = dt.now()
    t_secs = (dt_end - dt_start).total_seconds()
    vprint(  f"Finish: {dt_end.strftime('%H:%M, %a %d %b %Y')}"
           + f" (took {int(t_secs//60):02}:{int(t_secs%60):02})", verbose)


if __name__ == "__main__":
    main()
