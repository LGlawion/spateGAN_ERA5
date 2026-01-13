"""spateGAN-ERA5: Deep learning framework for ERA5 precipitation downscaling.

This package provides tools for spatio-temporal downscaling of ERA5
precipitation data from 24 km/1-hour to 2 km/10-minute resolution using
a probabilistic conditional GAN architecture.

Example usage:
    >>> from spategan_era5 import Generator, ERA5DownscalingInference
    >>> from spategan_era5.dataloader import load_and_prepare_dataset
    >>> from spategan_era5.projection import latlon_to_utm, utm_to_latlon
    >>> from spategan_era5.pipeline import run_downscaling_pipeline
"""

from src.spategan_era5.dataloader import (
    detect_cp_lsp_vars,
    load_and_prepare_dataset,
    normalize_longitude,
)
from src.spategan_era5.downscaling_inference import ERA5DownscalingInference
from src.spategan_era5.inference import InferenceEngine
from src.spategan_era5.model import Generator
from src.spategan_era5.pipeline import run_downscaling_pipeline
from src.spategan_era5.preprocessing import (
    calculate_domain_center,
    slice_data_for_projection,
    validate_patch_extraction,
    validate_time_dimension,
)
from src.spategan_era5.projection import (
    latlon_to_utm,
    prediction_output_dataset,
    utm_to_latlon,
)
from src.spategan_era5.utils import (
    DataInterpolation,
    generate_output_filename,
    haversine,
)

__all__ = [
    # Model
    "Generator",
    # Inference
    "InferenceEngine",
    "ERA5DownscalingInference",
    # Pipeline
    "run_downscaling_pipeline",
    # Data loading
    "load_and_prepare_dataset",
    "detect_cp_lsp_vars",
    "normalize_longitude",
    # Preprocessing
    "validate_patch_extraction",
    "slice_data_for_projection",
    "calculate_domain_center",
    "validate_time_dimension",
    # Projection
    "latlon_to_utm",
    "utm_to_latlon",
    "prediction_output_dataset",
    # Utilities
    "haversine",
    "DataInterpolation",
    "generate_output_filename",
]

__version__ = "0.1.0"
