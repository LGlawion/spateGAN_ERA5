#!/usr/bin/env python3
"""spateGAN-ERA5: ERA5 Precipitation Downscaling CLI.

Usage:
    uv run main.py --config config/config.yml
    uv run main.py --config config/config.yml --center-lat 50.0 --center-lon 10.0
    uv run main.py --help
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml

# Setup paths before imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spategan-era5",
        description="Downscale ERA5 precipitation to 2km/10min resolution",
    )
    
    parser.add_argument("-c", "--config", type=Path, default=Path("config/config.yml"),
                        help="Configuration YAML file")
    parser.add_argument("-i", "--input", type=str, help="Input ERA5 NetCDF file")
    parser.add_argument("--output-utm", type=str, help="Output directory for UTM")
    parser.add_argument("--output-latlon", type=str, help="Output directory for lat-lon")
    parser.add_argument("--center-lat", type=float, help="Center latitude (-90 to 90)")
    parser.add_argument("--center-lon", type=float, help="Center longitude (-180 to 180)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--stride", type=int, choices=range(1, 9), metavar="[1-8]",
                        help="Sliding window stride (hours)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    
    return parser


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments."""
    overrides = {
        ("data", "input_path"): args.input,
        ("data", "output_utm_path"): args.output_utm,
        ("data", "output_latlon_path"): args.output_latlon,
        ("domain", "center_lat"): args.center_lat,
        ("domain", "center_lon"): args.center_lon,
        ("time", "start_date"): args.start_date,
        ("time", "end_date"): args.end_date,
        ("processing", "seed"): args.seed,
        ("processing", "device"): args.device,
        ("inference", "stride_hours"): args.stride,
    }
    
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    
    return config


def main() -> int:
    """Main entry point."""
    # Parse CLI
    args = create_parser().parse_args()
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load and merge config
    try:
        config = load_config(PROJECT_ROOT / args.config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error("Configuration error: %s", e)
        return 1
    
    config = apply_cli_overrides(config, args)
    
    logger.info("Starting downscaling: center=(%.2f°N, %.2f°E), device=%s",
                config["domain"]["center_lat"],
                config["domain"]["center_lon"],
                config["processing"]["device"])
    
    # Run pipeline
    try:
        from src.spategan_era5.pipeline import run_downscaling_pipeline
        run_downscaling_pipeline(config, PROJECT_ROOT)
        return 0
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except ValueError as e:
        logger.error("Validation error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
