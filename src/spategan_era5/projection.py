"""
Projection and reprojection module for ERA5 data.

Handles conversion between lat-lon and UTM projections.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator




def get_utm_crs(utm_zone: int, center_lat: float) -> CRS:
    """
    Get the CRS (Coordinate Reference System) for UTM projection.
    
    Args:
        utm_zone: UTM zone number
        center_lat: Center latitude (to determine north/south)
    
    Returns:
        pyproj CRS object
    """
    # Determine hemisphere
    if center_lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    
    crs = CRS.from_epsg(epsg_code)
    return crs


def latlon_to_utm(ds_sliced, center_lat, center_lon, target_size, grid_spacing, method='bilinear'):
    """
    Transform xarray dataset from lat/lon to UTM projection.
    
    Parameters
    ----------
    ds_sliced : xarray.Dataset or xarray.DataArray
        Input dataset with dimensions (time, lat, lon) or (lat, lon)
    center_lat : float
        Center latitude (e.g., 22.6250)
    center_lon : float
        Center longitude (e.g., 96.3750)
    target_size : int
        Target grid size in pixels (e.g., 336 for 336x336 or 28 for 28x28)
    grid_spacing : float
        Grid spacing in meters (e.g., 2000 for 2km or 24000 for 24km)
    method : str
        Interpolation method: 'bilinear' (or 'linear') or 'nearest'
    
    Returns
    -------
    xarray.Dataset
        Transformed dataset with UTM coordinates
    """
    
    # Map method names
    if method == 'bilinear':
        method = 'linear'
    elif method not in ['linear', 'nearest']:
        raise ValueError(f"Unknown interpolation method: {method}. Use 'bilinear' or 'nearest'.")
    
    # Normalize center longitude to [-180, 180) so UTM zone calculation
    # and reprojection are consistent regardless of input lon convention.
    center_lon = ((center_lon + 180) % 360) - 180
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # Create CRS objects
    crs_latlon = CRS.from_epsg(4326)  # WGS84
    crs_utm = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    
    # Create transformer
    transformer_to_utm = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    transformer_to_latlon = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    
    # Handle DataArray input
    if isinstance(ds_sliced, xr.DataArray):
        var_name = ds_sliced.name or 'data'
        ds_sliced = ds_sliced.to_dataset(name=var_name)
    
    # Get original coordinates
    lats = ds_sliced.lat.values
    lons = ds_sliced.lon.values
    
    # Transform center point to UTM (center_lon already normalized)
    center_x, center_y = transformer_to_utm.transform(center_lon, center_lat)
    
    # Define target UTM grid (centered on center point)
    half_extent = (target_size * grid_spacing) / 2
    
    x_target = np.arange(
        center_x - half_extent + grid_spacing / 2,
        center_x + half_extent,
        grid_spacing
    )
    y_target = np.arange(
        center_y - half_extent + grid_spacing / 2,
        center_y + half_extent,
        grid_spacing
    )
    
    # Ensure exactly target_size pixels
    x_target = x_target[:target_size]
    y_target = y_target[:target_size]
    
    # Create target meshgrid
    x_target_grid, y_target_grid = np.meshgrid(x_target, y_target)
    
    # Transform target UTM grid back to lat/lon for interpolation
    lon_target, lat_target = transformer_to_latlon.transform(
        x_target_grid.ravel(), 
        y_target_grid.ravel()
    )
    lon_target = lon_target.reshape(x_target_grid.shape)
    lat_target = lat_target.reshape(y_target_grid.shape)
    
    # Get data variable names
    data_vars = list(ds_sliced.data_vars)
    
    # Check if time dimension exists
    has_time = 'time' in ds_sliced.dims
    
    # Prepare interpolator for vectorized operation
    data_dict = {}
    points = np.column_stack([lat_target.ravel(), lon_target.ravel()])
    
    for var in data_vars:
        if has_time:
            # Vectorized interpolation over time dimension
            data_3d = ds_sliced[var].values  # shape: (time, lat, lon)
            
            # Create interpolator for the entire 3D data
            interpolator = RegularGridInterpolator(
                (np.arange(data_3d.shape[0]), lats, lons),
                data_3d,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Create points array with time dimension (replicate for all times)
            n_times = len(ds_sliced.time)
            t_indices = np.repeat(np.arange(n_times), len(points))
            points_with_time = np.column_stack([
                t_indices,
                np.tile(points[:, 0], n_times),
                np.tile(points[:, 1], n_times)
            ])
            
            # Interpolate all times at once
            interpolated = interpolator(points_with_time).reshape(n_times, target_size, target_size)
            data_dict[var] = (['time', 'y', 'x'], interpolated)
        else:
            # Single time step (no time dimension)
            data = ds_sliced[var].values
            interpolator = RegularGridInterpolator(
                (lats, lons),
                data,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            interpolated = interpolator(points).reshape(target_size, target_size)
            data_dict[var] = (['y', 'x'], interpolated)
    
    coords = {
        'x': x_target,
        'y': y_target,
    }
    if has_time:
        coords['time'] = ds_sliced.time.values
    
    ds_utm = xr.Dataset(data_dict, coords=coords)
    
    # Add attributes
    ds_utm.attrs['crs'] = crs_utm.to_string()
    ds_utm.attrs['utm_zone'] = utm_zone
    ds_utm.attrs['center_lat'] = center_lat
    ds_utm.attrs['center_lon'] = center_lon
    ds_utm.attrs['grid_spacing_m'] = grid_spacing
    ds_utm.attrs['target_size'] = target_size
    ds_utm.x.attrs['units'] = 'meters'
    ds_utm.x.attrs['long_name'] = 'UTM easting'
    ds_utm.y.attrs['units'] = 'meters'
    ds_utm.y.attrs['long_name'] = 'UTM northing'
    
    # Add units to precipitation variables
    for var in data_vars:
        if 'cp' in var.lower() or 'lsp' in var.lower() or 'precip' in var.lower():
            ds_utm[var].attrs['units'] = 'mm/h'
            ds_utm[var].attrs['long_name'] = 'precipitation'
    
    return ds_utm




def utm_to_latlon(ds_utm, target_lats=None, target_lons=None, resolution=None, method='nearest'):
    """
    Transform xarray dataset from UTM projection back to lat/lon.
    
    Parameters
    ----------
    ds_utm : xarray.Dataset
        Input dataset with UTM coordinates (x, y) and necessary attributes
        (crs, center_lat, center_lon, utm_zone)
    target_lats : array-like, optional
        Target latitude coordinates. If None, will be computed from UTM extent.
    target_lons : array-like, optional
        Target longitude coordinates. If None, will be computed from UTM extent.
    resolution : float, optional
        Target resolution in degrees (e.g., 0.25 for 0.25° grid spacing).
        Used if target_lats/target_lons are not provided.
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns
    -------
    xarray.Dataset
        Transformed dataset with lat/lon coordinates
    """
    
    # Extract attributes from UTM dataset
    utm_zone = ds_utm.attrs.get('utm_zone')
    center_lat = ds_utm.attrs.get('center_lat')
    center_lon = ds_utm.attrs.get('center_lon')
    crs_string = ds_utm.attrs.get('crs')
    
    if utm_zone is None or center_lat is None:
        raise ValueError("Dataset missing required attributes (utm_zone, center_lat, center_lon)")
    
    # Determine hemisphere
    hemisphere = 'north' if center_lat >= 0 else 'south'
    
    # Create CRS objects
    crs_latlon = CRS.from_epsg(4326)  # WGS84
    if crs_string:
        crs_utm = CRS.from_string(crs_string)
    else:
        crs_utm = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    
    # Create transformers
    transformer_to_latlon = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
    transformer_to_utm = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
    
    # Get UTM coordinates
    x_utm = ds_utm.x.values
    y_utm = ds_utm.y.values
    
    # Transform all four corners and edges to get accurate lat/lon extent
    n_edge_points = 50
    
    # Sample along all edges
    edge_x_bottom = np.linspace(x_utm[0], x_utm[-1], n_edge_points)
    edge_y_bottom = np.full(n_edge_points, y_utm[0])
    edge_x_top = np.linspace(x_utm[0], x_utm[-1], n_edge_points)
    edge_y_top = np.full(n_edge_points, y_utm[-1])
    edge_x_left = np.full(n_edge_points, x_utm[0])
    edge_y_left = np.linspace(y_utm[0], y_utm[-1], n_edge_points)
    edge_x_right = np.full(n_edge_points, x_utm[-1])
    edge_y_right = np.linspace(y_utm[0], y_utm[-1], n_edge_points)
    
    all_edge_x = np.concatenate([edge_x_bottom, edge_x_top, edge_x_left, edge_x_right])
    all_edge_y = np.concatenate([edge_y_bottom, edge_y_top, edge_y_left, edge_y_right])
    
    all_edge_lon, all_edge_lat = transformer_to_latlon.transform(all_edge_x, all_edge_y)
    
    lat_min, lat_max = all_edge_lat.min(), all_edge_lat.max()
    lon_min, lon_max = all_edge_lon.min(), all_edge_lon.max()
    
    # Compute target lat/lon grid if not provided
    if target_lats is None or target_lons is None:
        if resolution is None:
            raise ValueError("Either provide target_lats/target_lons or specify resolution in degrees")
        
        if target_lats is None:
            # Snap to grid aligned with resolution
            lat_start = np.floor(lat_min / resolution) * resolution
            lat_end = np.ceil(lat_max / resolution) * resolution
            target_lats = np.arange(lat_start, lat_end + resolution / 2, resolution)
            # Trim to actual extent
            target_lats = target_lats[(target_lats >= lat_min) & (target_lats <= lat_max)]
        
        if target_lons is None:
            # Snap to grid aligned with resolution
            lon_start = np.floor(lon_min / resolution) * resolution
            lon_end = np.ceil(lon_max / resolution) * resolution
            target_lons = np.arange(lon_start, lon_end + resolution / 2, resolution)
            # Trim to actual extent
            target_lons = target_lons[(target_lons >= lon_min) & (target_lons <= lon_max)]
    
    # Create target lat/lon meshgrid
    lon_target, lat_target = np.meshgrid(target_lons, target_lats)
    
    # Transform target lat/lon to UTM for interpolation
    x_target, y_target = transformer_to_utm.transform(lon_target.ravel(), lat_target.ravel())
    x_target = x_target.reshape(lon_target.shape)
    y_target = y_target.reshape(lat_target.shape)
    
    # Get data variable names
    data_vars = list(ds_utm.data_vars)
    
    # Check if time dimension exists
    has_time = 'time' in ds_utm.dims
    
    # Prepare interpolator for vectorized operation
    data_dict = {}
    points = np.column_stack([y_target.ravel(), x_target.ravel()])
    
    for var in data_vars:
        if has_time:
            # Vectorized interpolation over time dimension
            data_3d = ds_utm[var].values  # shape: (time, y, x)
            
            # Create interpolator for the entire 3D data
            interpolator = RegularGridInterpolator(
                (np.arange(data_3d.shape[0]), y_utm, x_utm),
                data_3d,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Create points array with time dimension (replicate for all times)
            n_times = data_3d.shape[0]
            t_indices = np.repeat(np.arange(n_times), len(points))
            points_with_time = np.column_stack([
                t_indices,
                np.tile(points[:, 0], n_times),
                np.tile(points[:, 1], n_times)
            ])
            
            # Interpolate all times at once
            interpolated = interpolator(points_with_time).reshape(n_times, len(target_lats), len(target_lons))
            data_dict[var] = (['time', 'lat', 'lon'], interpolated)
        else:
            # Single time step (no time dimension)
            data = ds_utm[var].values
            interpolator = RegularGridInterpolator(
                (y_utm, x_utm),
                data,
                method=method,
                bounds_error=False,
                fill_value=np.nan
            )
            interpolated = interpolator(points).reshape(len(target_lats), len(target_lons))
            data_dict[var] = (['lat', 'lon'], interpolated)
    
    coords = {
        'lat': target_lats,
        'lon': target_lons,
    }
    if has_time:
        coords['time'] = ds_utm.time.values
    
    ds_latlon = xr.Dataset(data_dict, coords=coords)
    
    # Add attributes
    ds_latlon.attrs['crs'] = 'EPSG:4326'
    ds_latlon.attrs['resolution_deg'] = resolution
    ds_latlon.attrs['source_utm_zone'] = utm_zone
    ds_latlon.attrs['center_lat'] = center_lat
    ds_latlon.attrs['center_lon'] = center_lon
    ds_latlon.lat.attrs['units'] = 'degrees_north'
    ds_latlon.lat.attrs['long_name'] = 'latitude'
    ds_latlon.lon.attrs['units'] = 'degrees_east'
    ds_latlon.lon.attrs['long_name'] = 'longitude'
    
    # Add units to precipitation variables
    for var in data_vars:
        if 'precip' in var.lower():
            ds_latlon[var].attrs['units'] = 'mm/h'
            ds_latlon[var].attrs['long_name'] = 'precipitation'
    
    return ds_latlon




def prediction_output_dataset(ds):
    """
    Upsample xarray dataset from 1 hour to 10 minute resolution.
    Slice xarray to target domain.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset with hourly temporal resolution
  
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Upsampled dataset with 10 minute resolution
    """
    
    # xr.dataset for predictions:
    ds = ds.isel(x=np.arange(96, 168+72)).isel(y=np.arange(96, 168+72)) 
    ds = ds.drop_vars(list(ds.data_vars)[1]).rename({list(ds.data_vars)[0]: "precipitation"})
    ds["precipitation"][:] = np.nan
    
    
    n_times_orig = len(ds.time)
    
    ds_upsampled = ds.resample(time='10min')
    ds_upsampled = ds_upsampled.ffill()
  
    
    # Resample doesn't fill the last hour, so we need to extend manually
    times_orig = pd.DatetimeIndex(ds.time.values)
    times_new = pd.date_range(
        start=times_orig[0],
        end=times_orig[-1] + pd.Timedelta(minutes=50),
        freq='10min'
    )

    # Reindex to include the last 50 minutes
    ds_upsampled = ds_upsampled.reindex(time=times_new, method='ffill')

    # Sanity check
    n_times_new = len(ds_upsampled.time)
    expected_length = n_times_orig * 6
    
    if n_times_new != expected_length:
        raise ValueError(
            f"Length mismatch: new array has {n_times_new} steps, "
            f"expected {expected_length} (6 × {n_times_orig})"
        )
    
    return ds_upsampled
