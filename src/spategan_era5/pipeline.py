"""
Pipeline for ERA5 downscaling.

This module provides the main workflow functions that wire together
the dataloader, preprocessing, projection, and inference components.
"""

import logging
from pathlib import Path

import xarray as xr

from src.spategan_era5.dataloader import load_and_prepare_dataset
from src.spategan_era5.downscaling_inference import ERA5DownscalingInference
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
from src.spategan_era5.utils import generate_output_filename

logger = logging.getLogger(__name__)


def load_era5_data(
    input_path: Path,
    precip_unit: str = "m",
    required_hours: int = 16,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[xr.Dataset, str]:
    """Load and prepare ERA5 dataset.
    
    Args:
        input_path: Path to input NetCDF file.
        precip_unit: Unit of precipitation in input data.
        required_hours: Minimum required time steps in hours.
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        
    Returns:
        Tuple of (prepared dataset, variable name).
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If data validation fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info("Loading ERA5 data from %s", input_path)
    
    ds_era5, variable_name = load_and_prepare_dataset(
        str(input_path),
        precip_unit=precip_unit,
        precip_var=None,
        required_hours=required_hours,
        start_date=start_date,
        end_date=end_date,
    )
    
    logger.info("Loaded %d time steps: lat [%.1f, %.1f]°, lon [%.1f, %.1f]°",
                len(ds_era5.time),
                float(ds_era5.lat.min()),
                float(ds_era5.lat.max()),
                float(ds_era5.lon.min()),
                float(ds_era5.lon.max()))
    
    return ds_era5, variable_name


def extract_patch(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    patch_size_km: float,
    patch_padding_km: float,
    extra_padding_cells: int,
    required_hours: int,
) -> tuple[xr.Dataset, dict, tuple[float, float]]:
    """Validate domain and extract patch for processing.
    
    Args:
        ds: ERA5 dataset in lat-lon projection.
        center_lat: Center latitude of patch.
        center_lon: Center longitude of patch.
        patch_size_km: Size of patch in kilometers.
        patch_padding_km: Padding around patch in kilometers.
        extra_padding_cells: Extra cells for projection padding.
        required_hours: Minimum required hours of data.
        
    Returns:
        Tuple of (sliced dataset, slicing info, (center_lat, center_lon)).
        
    Raises:
        ValueError: If patch extraction validation fails.
    """
    logger.info("Validating patch extraction...")
    
    validate_patch_extraction(
        center_lat=center_lat,
        center_lon=center_lon,
        patch_size_km=672.0,
        patch_padding_km=100.0,
        lat_south=float(ds.lat.min().values),
        lat_north=float(ds.lat.max().values),
        lon_west=float(ds.lon.min().values),
        lon_east=float(ds.lon.max().values),
    )
    
    validate_time_dimension(ds, required_hours=required_hours)
    
    logger.info("Extracting patch centered at (%.4f°N, %.4f°E)...", center_lat, center_lon)
    
    ds_sliced, slicing_info = slice_data_for_projection(
        ds=ds,
        center_lat=center_lat,
        center_lon=center_lon,
        target_domain_size_km=672.0,
        extra_padding_cells=extra_padding_cells,
    )
    
    actual_center = calculate_domain_center(ds_sliced)
    
    return ds_sliced, slicing_info, actual_center


def project_to_utm(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Project data to UTM coordinates at two resolutions.
    
    Args:
        ds: Dataset in lat-lon projection.
        center_lat: Center latitude for UTM zone.
        center_lon: Center longitude for UTM zone.
        
    Returns:
        Tuple of (high-res 336x336 dataset, low-res 28x28 dataset).
    """
    ds_utm_336 = latlon_to_utm(
        ds, center_lat=center_lat, center_lon=center_lon,
        target_size=336, grid_spacing=2000, method="nearest",
    )
    
    ds_utm_28 = latlon_to_utm(
        ds, center_lat=center_lat, center_lon=center_lon,
        target_size=28, grid_spacing=24000, method="nearest",
    )
    
    return ds_utm_336, ds_utm_28


def run_inference(
    ds_utm_28: xr.Dataset,
    ds_utm_336: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    """Run model inference on prepared data.
    
    Args:
        ds_utm_28: Low-resolution UTM input dataset.
        ds_utm_336: High-resolution UTM dataset for output template.
        config: Configuration dictionary.
        
    Returns:
        Prediction dataset in UTM projection.
    """
    ds_utm_pred = prediction_output_dataset(ds_utm_336)
    
    inference = ERA5DownscalingInference(
        config=config,
        device=config["processing"]["device"],
        seed=config["processing"]["seed"],
    )
    
    x_tensor = inference.prepare_tensor_dataset(
        ds_utm_28, variable_names=list(ds_utm_28.data_vars)
    )
        
    logger.info("Running inference...")
    predictions = inference.predict_sliding_window(
        x_tensor,
        ds_prediction=ds_utm_pred,
        slide=config["inference"]["stride_hours"],
    )
    
    return predictions


def save_outputs(
    predictions_utm: xr.Dataset,
    ds_utm_28: xr.Dataset,
    output_utm_dir: Path | None,
    output_latlon_dir: Path | None,
    config: dict,
    save_model_input: bool = False,
) -> None:
    """Save prediction outputs to files.
    
    Args:
        predictions_utm: Predictions in UTM projection.
        ds_utm_28: Model input in UTM projection.
        output_utm_dir: Directory for UTM outputs.
        output_latlon_dir: Directory for lat-lon outputs.
        config: Configuration dictionary.
        save_model_input: Whether to save model input alongside predictions.
    """
    if output_latlon_dir:
        output_latlon_dir.mkdir(parents=True, exist_ok=True)
        
        predictions_latlon = utm_to_latlon(predictions_utm, resolution=0.018)
        
        filename = generate_output_filename(predictions_latlon, config, projection="latlon")
        predictions_latlon.to_netcdf(output_latlon_dir / filename)
        logger.info("Saved: %s", output_latlon_dir / filename)
        
        if save_model_input:
            ds_latlon = utm_to_latlon(ds_utm_28, resolution=0.25)
            filename_era = generate_output_filename(ds_latlon, config, projection="latlon", model="era5")
            ds_latlon.to_netcdf(output_latlon_dir / filename_era)
            logger.info("Saved: %s", output_latlon_dir / filename_era)
    
    if output_utm_dir:
        output_utm_dir.mkdir(parents=True, exist_ok=True)
        
        filename = generate_output_filename(predictions_utm, config, projection="utm")
        predictions_utm.to_netcdf(output_utm_dir / filename)
        logger.info("Saved: %s", output_utm_dir / filename)
        
        if save_model_input:
            filename_era = generate_output_filename(ds_utm_28, config, projection="utm", model="era5")
            ds_utm_28.to_netcdf(output_utm_dir / filename_era)
            logger.info("Saved: %s", output_utm_dir / filename_era)


def _create_precipitation_sums_plot(
    predictions_utm: xr.Dataset,
    ds_utm_28: xr.Dataset,
    project_root: Path,
    config: dict,
) -> None:
    """Create a simple side-by-side precipitation sums plot.

    Left: (ds_utm_28.cp + ds_utm_28.lsp).sum(dim='time')
    Right: predictions_utm.precipitation.sum(dim='time')

    Saves PNG to `project_root / plots` (or config['data']['plots_path']).
    """
    plotting_cfg = config.get("plotting", {})
    if not plotting_cfg.get("precipitation_sums", False):
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        left = None
        if ("cp" in ds_utm_28.data_vars) and ("lsp" in ds_utm_28.data_vars):
            left = (ds_utm_28["cp"] + ds_utm_28["lsp"]).sum(dim="time")
            left = left[8:-8, 8:-8]
        else:
            logger.warning("ds_utm_28 missing 'cp' and/or 'lsp' variables; left plot will be empty")

        right = None
        if "precipitation" in predictions_utm.data_vars:
            right = (predictions_utm["precipitation"]/6).sum(dim="time") # /6 since 6 time steps per hour
        else:
            logger.warning("predictions_utm missing 'precipitation' variable; right plot will be empty")

        if (left is None) and (right is None):
            logger.warning("No data available for precipitation sums plotting; skipping")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if left is not None:
            left.plot(ax=axes[0], cmap="turbo")
            axes[0].set_title("Input (target domain) precipitation sum (cp + lsp) - mm")
        else:
            axes[0].axis("off")

        if right is not None:
            right.plot(ax=axes[1], cmap="turbo")
            axes[1].set_title("Predicted precipitation sum - mm")
        else:
            axes[1].axis("off")

        plt.tight_layout()

        plots_dir = (
            project_root / config.get("data", {}).get("plots_path", "plots")
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_name = generate_output_filename(predictions_utm, config, projection="utm")
        plot_name = "precip_sums_" + plot_name.replace(".nc", ".png")
        out_path = plots_dir / plot_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        logger.info("Saved precipitation sums plot to %s", out_path)

    except Exception as e:
        logger.exception("Error creating precipitation sums plot: %s", e)


def fill_nans_if_sparse(ds: xr.Dataset, threshold=0.01) -> xr.Dataset:
    # total number of values across all variables
    total_values = sum(da.size for da in ds.data_vars.values())

    # total number of NaNs across all variables
    nan_values = sum(
        int(da.isnull().sum().values)
        for da in ds.data_vars.values()
    )

    nan_fraction = nan_values / total_values

    if nan_fraction < threshold and nan_fraction > 0:
        logger.info(f"Filling {nan_fraction:.4f} NaNs (below threshold {threshold:.4f})")
        return ds.fillna(0)
    elif nan_fraction >= threshold:
        logger.info(f"Not filling Nans since above threshold of {threshold:.4f}")
        assert nan_fraction < threshold, "Dataset too sparse with NaNs"
    else:
        return ds


def run_downscaling_pipeline(config: dict, project_root: Path) -> None:
    """Run the complete downscaling pipeline.
    
    This is the main orchestration function that wires together all
    pipeline stages: load -> extract -> project -> infer -> save.
    
    Args:
        config: Configuration dictionary.
        project_root: Project root directory.
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If validation fails.
    """
    # Stage 1: Load data
    input_path = project_root / config["data"]["input_path"]
    ds_era5, _ = load_era5_data(
        input_path=input_path,
        precip_unit=config["data"].get("precip_unit", "m"),
        required_hours=16,
        start_date=config["time"].get("start_date"),
        end_date=config["time"].get("end_date"),
    )
    
    # Stage 2: Extract patch
    ds_sliced, _, center_coords = extract_patch(
        ds=ds_era5,
        center_lat=config["domain"]["center_lat"],
        center_lon=config["domain"]["center_lon"],
        patch_size_km=672.0,
        patch_padding_km=100.0,
        extra_padding_cells=config["projection"]["extra_padding_cells"],
        required_hours=16,
    )
    
    logger.info("running UTM projection")
    
    # Stage 3: Project to UTM
    ds_utm_336, ds_utm_28 = project_to_utm(ds_sliced, *center_coords)
    
    # Stage 3.1: Fill NaNs if sparse
    ds_utm_28 = fill_nans_if_sparse(ds_utm_28, threshold=config.get("data", {}).get("nan_fill_threshold", 0.01))
        
    # Stage 4: Run inference
    predictions_utm = run_inference(ds_utm_28, ds_utm_336, config)
    
    # Stage 4: Plot precipitation maps (optional)
    
    # Stage 5: Save outputs
    output_utm_dir = (
        project_root / config["data"]["output_utm_path"]
        if config["data"].get("output_utm_path") else None
    )
    output_latlon_dir = (
        project_root / config["data"]["output_latlon_path"]
        if config["data"].get("output_latlon_path") else None
    )
    
    save_outputs(
        predictions_utm=predictions_utm,
        ds_utm_28=ds_utm_28,
        output_utm_dir=output_utm_dir,
        output_latlon_dir=output_latlon_dir,
        config=config,
        save_model_input=config["data"].get("save_model_input", False),
    )

    # Optional simple plotting
    try:
        _create_precipitation_sums_plot(predictions_utm, ds_utm_28, project_root, config)
    except Exception:
        # _create_precipitation_sums_plot logs its own errors
        pass
    
    logger.info("Downscaling pipeline completed successfully!")
