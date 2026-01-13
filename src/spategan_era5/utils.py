"""
Utility functions for spateGAN-ERA5.

Contains helper functions for distance calculations, interpolation,
and filename generation.
"""

import math
from pathlib import Path

import pandas as pd
import torch.nn as nn

# Earth radius in kilometers (WGS84 mean radius)
EARTH_RADIUS_KM = 6371.04


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula for accurate distance calculation on a sphere.
    
    Args:
        lat1: Latitude of first point in degrees.
        lon1: Longitude of first point in degrees.
        lat2: Latitude of second point in degrees.
        lon2: Longitude of second point in degrees.
        
    Returns:
        Distance between the points in kilometers.
    """

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = EARTH_RADIUS_KM * c

    return distance



class DataInterpolation(nn.Module):
    """PyTorch module for interpolating 5D tensor data.
    
    Handles both 2D (bicubic/bilinear/nearest) and 3D (trilinear) interpolation
    modes for batch, channel, time, height, width tensors.
    
    Args:
        size: Target output size (height, width) or (time, height, width).
        mode: Interpolation mode ('bicubic', 'bilinear', 'nearest', 'trilinear').
        corners: Whether to align corners in interpolation.
        antialias: Whether to apply antialiasing.
    """
    
    def __init__(
        self,
        size: tuple[int, int] | tuple[int, int, int],
        mode: str,
        corners: bool | None = None,
        antialias: bool | None = None,
    ) -> None:
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = corners
        self.antialias = antialias

    def forward(self, x):
        """Interpolate the input tensor.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W).
            
        Returns:
            Interpolated tensor.
        """
        if self.mode in ('bicubic', 'bilinear', 'nearest'):
            b, c, t, w, h = x.size()
            x = x.reshape(b, c * t, w, h)
            x = self.interp(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                antialias=self.antialias,
            )
            x = x.reshape(b, c, t, int(self.size[0]), int(self.size[1]))
        else:
            x = self.interp(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                antialias=self.antialias,
            )
        return x
    

def generate_output_filename(
    dataset: "xr.Dataset",
    config: dict,
    projection: str = 'latlon',
    model: str = 'spateGAN_ERA5',
) -> str:
    """
    Generate output filename based on parameters.
    
    Parameters
    ----------
    dataset: xr.Dataset
        Dataset containing time and coordinate information
    projection : str
        Projection type ('latlon' or 'utm')
    
    Returns
    -------
    str
        Formatted filename
    """
    
    center_lat = dataset.attrs['center_lat']
    center_lon = dataset.attrs['center_lon'] 
    start_date = dataset.time.values[0]
    end_date = dataset.time.values[-1]

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Format lat/lon with sign
    lat_str = f"{abs(center_lat):.2f}{'N' if center_lat >= 0 else 'S'}"
    lon_str = f"{abs(center_lon):.2f}{'E' if center_lon >= 0 else 'W'}"
    
    # Format dates
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    if model == 'spateGAN_ERA5':
        filename = f"{model}_{projection}_{lat_str}_{lon_str}_{start_str}_{end_str}_e{config['processing']['seed']}.nc"
    elif model == 'era5':
        filename = f"{model}_{projection}_{lat_str}_{lon_str}_{start_str}_{end_str}.nc"
    else:
        raise ValueError(f"Unknown model type: {model}")
    
    return filename