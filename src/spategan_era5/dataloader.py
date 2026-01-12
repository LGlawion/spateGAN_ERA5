"""
Data loader utilities for ERA5 datasets.

Provides functions to load datasets, normalize coordinate names, and convert
precipitation units to `mm/hr`.
"""

import numpy as np
import xarray as xr


# Precipitation variable name candidates
CP_CANDIDATES = ['cp', 'convective', 'convcp', 'conv_precip', 'conv_precipitation']
LSP_CANDIDATES = ['lsp', 'ls_precip', 'large_scale', 'ls_precipitation', 'stratiform']


def detect_cp_lsp_vars(ds: xr.Dataset) -> list[str]:
    """
    Detect CP (convective) and LSP (large-scale) precipitation variable names.
    
    Searches the dataset for variables matching common naming patterns for
    convective and large-scale precipitation.
    
    Args:
        ds: xarray Dataset to search.
        
    Returns:
        List with found variables in order [CP, LSP].
        
    Raises:
        ValueError: If CP or LSP variables cannot be found.
    """
    found_cp: str | None = None
    found_lsp: str | None = None

    for var in ds.data_vars:
        name = var.lower()
        if found_cp is None and any(c in name for c in CP_CANDIDATES):
            found_cp = var
        if found_lsp is None and any(c in name for c in LSP_CANDIDATES):
            found_lsp = var

    # Also accept exact uppercase names 
    if found_cp is None and 'CP' in ds.data_vars:
        found_cp = 'CP'
    if found_lsp is None and 'LSP' in ds.data_vars:
        found_lsp = 'LSP'

    # Check if both variables were found
    available_vars = list(ds.data_vars)
    
    if found_cp is None and found_lsp is None:
        raise ValueError(
            f"Could not find CP (convective) or LSP (large-scale) precipitation variables. "
            f"Available variables: {available_vars}. "
            f"Expected CP names: {CP_CANDIDATES + ['CP']}. "
            f"Expected LSP names: {LSP_CANDIDATES + ['LSP']}."
        )
    elif found_cp is None:
        raise ValueError(
            f"Could not find CP (convective) precipitation variable. "
            f"Available variables: {available_vars}. "
            f"Expected names: {CP_CANDIDATES + ['CP']}."
        )
    elif found_lsp is None:
        raise ValueError(
            f"Could not find LSP (large-scale) precipitation variable. "
            f"Available variables: {available_vars}. "
            f"Expected names: {LSP_CANDIDATES + ['LSP']}."
        )
    
    return [found_cp, found_lsp]


# Unit conversion factors
UNIT_CONVERSIONS: dict[str, float] = {
    'mm/hr': 1.0,
    'mm/h': 1.0,
    'mmperhr': 1.0,
    'mm/s': 3600.0,
    'mmsec': 3600.0,
    'm/s': 1000.0 * 3600.0,
    'msec': 3600.0 * 1000.0,
    'mpersec': 3600.0 * 1000.0,
    'm/hr': 1000.0,
    'm/h': 1000.0,
    'mperhr': 1000.0,
    'm': 1000.0,  # assume m per hour (ERA5 tp accumulated)
}


def _conversion_factor_to_mm_per_hr(unit: str) -> float:
    """
    Compute multiplicative factor to convert given unit to mm/hr.
    
    Args:
        unit: Input unit string (e.g., 'm/s', 'm/hr', 'mm/hr').
        
    Returns:
        Multiplication factor to convert to mm/hr.
    """
    normalized_unit = unit.strip().lower().replace(' ', '')
    return UNIT_CONVERSIONS.get(normalized_unit, 1000.0)  # default fallback


def normalize_longitude(
    ds: xr.Dataset
) -> xr.Dataset:
    """
    Normalize longitude coordinates to -180..180 range.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with longitude coordinate
   
    Returns
    -------
    xarray.Dataset
        Dataset with longitude normalized to -180..180
    """
    if 'lon' not in ds.coords:
        return ds
    
    lons = ds['lon'].values
    lon_min = np.nanmin(lons)
    lon_max = np.nanmax(lons)
    
    if lon_min < 0:
        # Contains negative values -> must be -180..180
        is_0_360 = False
    elif lon_max > 180:
        # Values exceed 180 -> must be 0..360
        is_0_360 = True
    else:
        # Ambiguous range, assume -180..180
        is_0_360 = False
    
    # Convert if needed
    if is_0_360:
        # Convert 0..360 to -180..180
        new_lons = np.where(lons > 180, lons - 360, lons)
        ds = ds.assign_coords(lon=new_lons)
        ds = ds.sortby('lon')
    else:
        # Already -180..180, just ensure sorted
        if not np.all(np.diff(lons) > 0):
            ds = ds.sortby('lon')
    
    return ds



def load_and_prepare_dataset(
    path: str,
    precip_unit: str = 'm',
    precip_var: str | None = None,
    required_hours: int = 16,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[xr.Dataset, str]:
    """
    Load ERA5 dataset from `path`, normalize coordinate names, optionally filter
    by date range, select the first `required_hours` hours (or within date range),
    detect precipitation variable (unless provided), and convert its units to `mm/hr`.

    Parameters:
        path: File path to netCDF dataset
        precip_unit: Unit of precipitation in input data
        precip_var: Specific precipitation variable name (auto-detect if None)
        required_hours: Number of hours to extract
        start_date: Start date for filtering (format: 'YYYY-MM-DD', optional)
        end_date: End date for filtering (format: 'YYYY-MM-DD', optional)

    Returns:
        ds_prepared, precip_var_name
    """
    ds = xr.open_dataset(path)

    # Rename dimension names
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    if 'latitude' in ds.dims:
        ds = ds.rename({'latitude': 'lat'})
    if 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon'})

    # Normalize longitude
    ds = normalize_longitude(ds)

    # Detect CP/LSP variables (prefer CP and LSP channels)
    precip_vars = detect_cp_lsp_vars(ds)

    # Ensure time dimension exists
    if 'time' not in ds.dims:
        raise ValueError("Dataset must contain a 'time' dimension")

    # Filter by date range if provided
    if start_date is not None or end_date is not None:
        ds = ds.sel(time=slice(start_date, end_date))
        if len(ds.time) == 0:
            raise ValueError(f"No data found in the specified date range ({start_date} to {end_date})")
    
    # Slice to required hours if available
    if len(ds.time) < required_hours:
        raise ValueError(f"Dataset has {len(ds.time)} time steps but {required_hours} are required")

    # Keep only the detected precipitation variables (drop everything else)
    # order to CP, LSP
    ds = ds[[v for v in precip_vars if v in ds.data_vars]]

    # Convert precipitation units to mm/hr for each precip var
    factor = _conversion_factor_to_mm_per_hr(precip_unit)
    for v in list(ds.data_vars):
        ds[v] = ds[v] * factor
        ds[v].attrs['units'] = 'mm/hr'

    # attach detected variable list for callers that want both channels
    ds.attrs['precip_variables'] = precip_vars

    # return the primary variable name (first detected)
    primary = precip_vars[0] if precip_vars else list(ds.data_vars)[0]
    return ds, primary
