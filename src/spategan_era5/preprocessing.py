"""
Preprocessing module for ERA5 data validation and slicing.

Handles domain size validation and data extraction.
"""

import numpy as np
import xarray as xr
from typing import Tuple

from src.spategan_era5.utils import haversine


def validate_patch_extraction(
    center_lat: float,
    center_lon: float,
    patch_size_km: float,
    patch_padding_km: float,
    lat_south: float,
    lat_north: float,
    lon_west: float,
    lon_east: float,
) -> dict[str, float]:
    """
    Validate that the data domain is large enough to extract the requested patch.
    
    The patch is extracted centered at (center_lat, center_lon) with a radius of
    (patch_size_km/2 + patch_padding_km) in all directions to account for lat-lon
    distortion and ensure coverage after UTM projection.
    
    Args:
        center_lat: Extraction center latitude (degrees)
        center_lon: Extraction center longitude (degrees)
        patch_size_km: Base patch size in km (e.g., 672)
        patch_padding_km: Extra padding for distortion (e.g., 100)
        lat_south: Southern boundary of available data (degrees)
        lat_north: Northern boundary of available data (degrees)
        lon_west: Western boundary of available data (degrees)
        lon_east: Eastern boundary of available data (degrees)
    
    Returns:
        Dictionary with validation results
    
    Raises:
        ValueError: If patch cannot be extracted from available data
    """
    # Calculate required radius from center
    half_patch_km = patch_size_km / 2.0
    required_radius_km = half_patch_km + patch_padding_km
    
    errors = []
    
    # First check if center is within data domain
    if not (lat_south <= center_lat <= lat_north):
        errors.append(f"Center latitude {center_lat:.2f}° is outside data range [{lat_south:.2f}°, {lat_north:.2f}°]")
    
    # For longitude, handle wraparound for global data
    # Normalize to same convention for comparison
    center_lon_norm = ((center_lon + 180) % 360) - 180
    lon_west_norm = ((lon_west + 180) % 360) - 180
    lon_east_norm = ((lon_east + 180) % 360) - 180
    
    # Check if it's global longitude data
    is_global_lon = abs(lon_east - lon_west) > 350
    
    if not is_global_lon:
        # Regional data: center must be within bounds
        if lon_west_norm <= lon_east_norm:
            # Normal case
            if not (lon_west_norm <= center_lon_norm <= lon_east_norm):
                errors.append(f"Center longitude {center_lon:.2f}° is outside data range [{lon_west:.2f}°, {lon_east:.2f}°]")
        else:
            # Wraps around 180° meridian
            if not (center_lon_norm >= lon_west_norm or center_lon_norm <= lon_east_norm):
                errors.append(f"Center longitude {center_lon:.2f}° is outside data range [{lon_west:.2f}°, {lon_east:.2f}°]")
    
    if center_lon > 180:
        errors.append(f"Center longitude {center_lon:.2f}° exceeds 180°. Please use -180 to 180 range.")
    
    # If center is outside data, fail immediately
    if errors:
        error_msg = "Cannot extract patch:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    # Check distance coverage from center to data boundaries
    lat_distance_south = haversine(center_lat, 0, lat_south, 0)
    lat_distance_north = haversine(center_lat, 0, lat_north, 0)
    
    # Check longitude coverage (at center latitude)
    lon_distance_west = haversine(center_lat, center_lon, center_lat, lon_west)
    lon_distance_east = haversine(center_lat, center_lon, center_lat, lon_east)
    
    # Validate coverage
    if lat_distance_south < required_radius_km:
        errors.append(f"Insufficient coverage south: {lat_distance_south:.2f} km < {required_radius_km:.2f} km")
    if lat_distance_north < required_radius_km:
        errors.append(f"Insufficient coverage north: {lat_distance_north:.2f} km < {required_radius_km:.2f} km")
    if lon_distance_west < required_radius_km:
        errors.append(f"Insufficient coverage west: {lon_distance_west:.2f} km < {required_radius_km:.2f} km")
    if lon_distance_east < required_radius_km:
        errors.append(f"Insufficient coverage east: {lon_distance_east:.2f} km < {required_radius_km:.2f} km")
    
    if errors:
        error_msg = "Cannot extract patch:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)
    
    print(f"✓ Patch validated: center ({center_lat:.2f}°N, {center_lon:.2f}°E), radius {required_radius_km:.0f} km")
    
    return {
        "lat_distance_south": lat_distance_south,
        "lat_distance_north": lat_distance_north,
        "lon_distance_west": lon_distance_west,
        "lon_distance_east": lon_distance_east,
        "required_radius_km": required_radius_km
    }


def slice_data_for_projection(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    target_domain_size_km: float,
    extra_padding_cells: int = 2,
    era5_grid_size_km: float = 25.0,
) -> tuple[xr.Dataset, dict]:
    """
    Slice ERA5 data to extract a centered patch in lat-lon projection.
    
    Extracts a patch centered on the specified coordinates that is slightly 
    larger than the target domain size (672×672 km). This is done in lat-lon 
    projection before any UTM conversion.
    
    Uses Haversine distance to calculate required pixels accounting for
    latitude-dependent longitude spacing. Handles global data with longitude
    wrapping (0-360 or -180-180).
    
    Strategy:
    1. Calculate actual lat/lon spacing using Haversine at center coordinates
    2. For latitude: use spacing at center_lat
    3. For longitude: use spacing at center_lat (longitude compression applies at any latitude)
    4. Add padding: pixels = ceil(domain_km/2 / spacing) * 2 + padding
    5. Extract rectangular patch centered at (center_lat, center_lon)
    
    Args:
        ds: Input xarray Dataset with lat/lon coordinates
        center_lat: Center latitude for extraction (degrees)
        center_lon: Center longitude for extraction (degrees)
        target_domain_size_km: Target domain size in km (e.g., 672)
        extra_padding_cells: Extra cells for padding (default 2)
        era5_grid_size_km: ERA5 baseline grid resolution in km (~25 km) - used as fallback
    
    Returns:
        Tuple of (sliced_dataset, slicing_info_dict)
        - sliced_dataset: xr.Dataset with shape (N_lat, N_lon, time)
        - slicing_info_dict: Dict with metadata about the extraction
    """
    # Get coordinate arrays
    lat_coords = ds.coords['lat'].values
    lon_coords = ds.coords['lon'].values
    
    # Find indices closest to center coordinates
    center_lat_idx = np.argmin(np.abs(lat_coords - center_lat))
    center_lon_idx = np.argmin(np.abs(lon_coords - center_lon))
    
    # Half-domain size in km
    half_domain_km = target_domain_size_km / 2.0
    
    # ===== LATITUDE SPACING =====
    # Calculate actual spacing between consecutive latitude indices at center
    if center_lat_idx > 0 and center_lat_idx < len(lat_coords) - 1:
        lat1 = lat_coords[center_lat_idx]
        lat2 = lat_coords[center_lat_idx - 1]
        lat_spacing_km = haversine(lat1, 0, lat2, 0)
    else:
        # Fallback: use the spacing at available adjacent index
        if center_lat_idx < len(lat_coords) - 1:
            lat1 = lat_coords[center_lat_idx]
            lat2 = lat_coords[center_lat_idx + 1]
            lat_spacing_km = haversine(lat1, 0, lat2, 0)
        else:
            lat_spacing_km = era5_grid_size_km
    
    # Latitude pixels needed (in each direction from center)
    lat_pixels_half = int(np.ceil(half_domain_km / lat_spacing_km))
    # padding applies to each side (top and bottom)
    lat_pixels_total = lat_pixels_half * 2 + extra_padding_cells * 2
    
    # ===== LONGITUDE SPACING =====
    # Calculate actual spacing between consecutive longitude indices at center_lat
    # (longitude spacing depends on latitude: cos(lat) factor)
    if center_lon_idx > 0 and center_lon_idx < len(lon_coords) - 1:
        lon1 = lon_coords[center_lon_idx]
        lon2 = lon_coords[center_lon_idx - 1]
        lon_spacing_km = haversine(center_lat, lon1, center_lat, lon2)
    else:
        # Fallback: use spacing at available adjacent index
        if center_lon_idx < len(lon_coords) - 1:
            lon1 = lon_coords[center_lon_idx]
            lon2 = lon_coords[center_lon_idx + 1]
            lon_spacing_km = haversine(center_lat, lon1, center_lat, lon2)
        else:
            lon_spacing_km = era5_grid_size_km * np.cos(np.radians(center_lat))
    
    # Ensure we don't divide by zero for polar regions
    if lon_spacing_km < 1e-6:
        lon_spacing_km = 0.1  # Near pole, use a small positive value
    
    # Longitude pixels needed (in each direction from center)
    lon_pixels_half = int(np.ceil(half_domain_km / lon_spacing_km))
    # padding applies to each side (left and right)
    lon_pixels_total = lon_pixels_half * 2 + extra_padding_cells * 2
    
    # ===== EXTRACT PATCH =====
    # Calculate slice indices for latitude (straightforward)
    # apply padding on both sides
    lat_idx_start = center_lat_idx - lat_pixels_half - extra_padding_cells
    lat_idx_end = center_lat_idx + lat_pixels_half + extra_padding_cells + (lat_pixels_total % 2)
    
    # Clamp latitude to valid range
    lat_idx_start = max(0, lat_idx_start)
    lat_idx_end = min(len(lat_coords), lat_idx_end)
    
    # For longitude, handle global wrapping (for data with lon 0-360 or -180-180)
    is_global_lon = (len(lon_coords) > 100) and (lon_coords[-1] - lon_coords[0] > 350)
    
    if is_global_lon:
        # Global data: allow wrapping around and apply padding on both sides
        lon_idx_start = (center_lon_idx - lon_pixels_half - extra_padding_cells) % len(lon_coords)
        lon_idx_end = (center_lon_idx + lon_pixels_half + extra_padding_cells + (lon_pixels_total % 2)) % len(lon_coords)
        
        if lon_idx_end < lon_idx_start:
            # Wrapping around the dateline
            ds_sliced = xr.concat([
                ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                       lon=slice(lon_idx_start, None)),
                ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                       lon=slice(None, lon_idx_end))
            ], dim='lon')
        else:
            ds_sliced = ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                               lon=slice(lon_idx_start, lon_idx_end))
    else:
        # Limited longitude range: clamp without wrapping
        lon_idx_start = max(0, center_lon_idx - lon_pixels_half - extra_padding_cells)
        lon_idx_end = min(len(lon_coords), center_lon_idx + lon_pixels_half + extra_padding_cells + (lon_pixels_total % 2))
        
        ds_sliced = ds.isel(lat=slice(lat_idx_start, lat_idx_end),
                           lon=slice(lon_idx_start, lon_idx_end))
    
    # Compile slicing information
    slicing_info = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_lat_idx": center_lat_idx,
        "center_lon_idx": center_lon_idx,
        "lat_indices": (lat_idx_start, lat_idx_end),
        "lon_indices": (lon_idx_start, lon_idx_end) if not is_global_lon or lon_idx_end >= lon_idx_start else "wrapped",
        "extracted_lat_range": (float(ds_sliced.lat.min()), float(ds_sliced.lat.max())),
        "extracted_lon_range": (float(ds_sliced.lon.min()), float(ds_sliced.lon.max())),
        "extracted_shape": (len(ds_sliced.lat), len(ds_sliced.lon)),
        "lat_pixels_total": lat_pixels_total,
        "lon_pixels_total": lon_pixels_total,
        "lat_spacing_km": lat_spacing_km,
        "lon_spacing_km": lon_spacing_km,
        "target_domain_size_km": target_domain_size_km,
        "is_global_data": is_global_lon,
    }
    
    return ds_sliced, slicing_info


def calculate_domain_center(ds: xr.Dataset) -> Tuple[float, float]:
    """
    Calculate the center coordinates of the sliced domain.
    
    Uses the geographic center of the provided dataset coordinates.
    
    Args:
        ds: xarray Dataset with lat/lon coordinates (typically sliced data)
    
    Returns:
        Tuple of (center_lat, center_lon)
    """
    lat_coords = ds.coords['lat'].values
    lon_coords = ds.coords['lon'].values
    
    center_lat = (lat_coords.min() + lat_coords.max()) / 2.0
    center_lon = (lon_coords.min() + lon_coords.max()) / 2.0
    
    return center_lat, center_lon


def validate_time_dimension(ds: xr.Dataset, required_hours: int = 16) -> bool:
    """
    Validate that the dataset has enough time steps.
    
    For a 16-hour requirement, we need 16 hourly time steps.
    
    Args:
        ds: Input dataset
        required_hours: Number of required hours
    
    Returns:
        True if valid, raises ValueError otherwise
    
    Raises:
        ValueError: If insufficient time steps
    """
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension")
    
    n_time_steps = len(ds.time)
    
    if n_time_steps < required_hours:
        raise ValueError(
            f"Dataset has {n_time_steps} time steps but {required_hours} are required"
        )
    
    # Warning for large datasets (>1 month ~ 720 hours)
    if n_time_steps > 720:
        print(f"⚠ Warning: Large dataset with {n_time_steps} time steps (~{n_time_steps/24:.1f} days) may cause memory issues")
    
    return True


def extract_time_window(
    ds: xr.Dataset,
    start_idx: int,
    n_hours: int
) -> xr.Dataset:
    """
    Extract a specific time window from the dataset.
    
    Args:
        ds: Input dataset
        start_idx: Starting time index
        n_hours: Number of hours to extract
    
    Returns:
        Sliced dataset with the specified time window
    """
    return ds.isel(time=slice(start_idx, start_idx + n_hours))
