#!/usr/bin/env python3
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import xarray as xr
import cdsapi
from ecmwf.opendata import Client

AIFS_OUT_DIR = "./data/aifs"
ERA5_TARGET_PATH = Path("data/era5_input.nc")

ECMWF_FORECAST_STEPS = list(range(0, 32, 6))

def prepare_ecmwf_data(out_dir: str) -> Path:
    os.makedirs(out_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    raw_filename = os.path.join(out_dir, f"ecwmf_{date}_aifs.grib2")

    clean_filename = Path(out_dir) / f"ecwmf_{date}_clean.nc"

    client = Client(source="ecmwf", model="aifs-single")
    client.retrieve(
        step=ECMWF_FORECAST_STEPS,
        type="fc",
        param=["cp", "tp"],
        target=raw_filename,
    )

    # You may need the 'cfgrib' engine installed
    ds = xr.open_dataset(raw_filename, engine="cfgrib")

    ds = xr.open_dataset(raw_filename)
    ds = ds.rename({
        'valid_time': 'time',
        'time': 'start_time',
    })
    ds = ds.swap_dims({'step': 'time'})
    ds['lsp'] = ds['tp'] - ds['cp']

    ds_hourly = ds.resample(time="1h").interpolate(kind='linear')
    ds_hourly_rate = ds_hourly.diff(dim="time")

    print(f"Units: {ds['tp'].attrs.get('units')}")

    if ds['tp'].attrs.get('units') == 'kg m**-2':
        ds_mm = ds_hourly_rate / 1000
    else:
        ds_mm = ds_hourly_rate
    ds_spateGAN = ds_mm[['cp', 'lsp']]

    ds_spateGAN.to_netcdf(clean_filename)

    return clean_filename

def download_era5_cds(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'convective_precipitation',
                'large_scale_precipitation',
            ],
            'year': '2026',
            'month': '03',
            'day': '01',
            'time': [f"{str(hour).zfill(2)}:00" for hour in range(24)],
            'area': [54.0, 5.0, 50.0, 9.0], # North, West, South, East
        },
        str(out_path)
    )
    return out_path

def execute_downscaling(input_file: Path):
    command = [
        sys.executable, "main.py",
        "--input", str(input_file),
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    processed_file = prepare_ecmwf_data(AIFS_OUT_DIR)

    # download_era5_cds(ERA5_TARGET_PATH)
    execute_downscaling(processed_file)
