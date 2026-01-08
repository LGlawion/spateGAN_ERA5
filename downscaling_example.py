"""
spateGAN-ERA5 spatiotemporal precipitation downscaling example
https://www.nature.com/articles/s41612-025-01103-y

This script runs inference on test example data and saves visualization plots.
Run with: uv run downscale.py
"""

import torch
import xarray as xr
import numpy as np
from einops import rearrange
import pathlib
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.spategan_era5.model import Generator
from src.spategan_era5.inference import InferenceEngine


def main():
    # Setup paths
    project_root = pathlib.Path(__file__).parent
    model_path = project_root / "model_weights" / "model_weights.pt"
    fn_test_y = project_root / "data" / "y_test.nc"
    fn_test_x = project_root / "data" / "x_test.nc"
    
    # Create output directory for plots
    plot_dir = project_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Verify required files exist
    for file_path, file_name in [
        (model_path, "model weights"),
        (fn_test_x, "test input data"),
        (fn_test_y, "test target data"),
    ]:
        if not file_path.exists():
            print(f"ERROR: {file_name} not found at {file_path}", file=sys.stderr)
            return 1

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # Load model and weights
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
    try:
        spateGAN_era5 = Generator().to(device)
        checkpoint = torch.load(model_path, weights_only=True)
        spateGAN_era5.load_state_dict(checkpoint, strict=True)
        spateGAN_era5.eval()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        return 1

    # Load example dataset
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading test datasets...")
    try:
        ds_test_y = xr.open_dataset(fn_test_y).load()
        ds_test_x = xr.open_dataset(fn_test_x).load()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}", file=sys.stderr)
        return 1

    # Prepare data
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing data...")
    try:
        x_test = np.stack([ds_test_x.cp.values, ds_test_x.lsp.values])
        x_test = rearrange(x_test, "c t h w -> 1 c t h w")
        y_test = rearrange(ds_test_y.rainfall_amount.values, "t h w -> 1 1 t h w")
        
        # Validate data shapes and ranges
        assert x_test.shape[1] == 2, f"Expected 2 input channels, got {x_test.shape[1]}"
        assert y_test.shape[1] == 1, f"Expected 1 target channel, got {y_test.shape[1]}"
        assert not np.isnan(x_test).any(), "Input data contains NaN values"
        assert not np.isnan(y_test).any(), "Target data contains NaN values"
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data validation passed.")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Target shape: {y_test.shape}")
    except Exception as e:
        print(f"ERROR: Failed to prepare data: {e}", file=sys.stderr)
        return 1

    # Downscale data to 2x2km & 10 min resolution
    # Patchsizes:
    # x: (batch, channels, time, width, height) = (batch, 2, 16, 28, 28) = (batch, CP & LSP, 16hr, 672km, 672km)
    # y: (batch, channels, time, width, height) = (batch, 1, 48, 168, 168) = (batch,TP, 8hr, 336km, 336km) --> cropped to (batch,TP, 6hr, 288km, 288km)

    # Initialize InferenceEngine
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running inference...")
    try:
        engine = InferenceEngine(spateGAN_era5)
        prediction, target, era5 = engine.infer(x_test, target=y_test, seed=4)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inference completed successfully.")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  ERA5 shape: {era5.shape}")
    except Exception as e:
        print(f"ERROR: Inference failed: {e}", file=sys.stderr)
        return 1

    # Plot results
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating visualization...")
    try:
        timestep = 100
        vmax = 20
        vmin = 0.01

        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(3, 7, figure=fig, width_ratios=[1] * 6 + [0.05])

        # Plot ERA5 TP data
        ax = fig.add_subplot(gs[0, 0])
        img = ax.imshow(era5[timestep // 6], cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(f"ERA5 TP, t")
        ax.axis("off")

        axes_tar = []
        for j in range(6):
            ax = fig.add_subplot(gs[1, j])
            axes_tar.append(ax)

        axes_pred = []
        for j in range(6):
            ax = fig.add_subplot(gs[2, j])
            axes_pred.append(ax)

        # Plot RADKLIM-YW data
        for i, ax in enumerate(axes_tar):
            img = ax.imshow(target[timestep + i], cmap="turbo", vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(f"RADKLIM-YW t+{i*10}min.")
            else:
                ax.set_title(f"t+{i*10}min.")
            ax.axis("off")

        # Plot predictions
        for i, ax in enumerate(axes_pred):
            img = ax.imshow(prediction[timestep + i], cmap="turbo", vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(f"spateGAN-ERA5 t+{i*10}min.")
            else:
                ax.set_title(f"t+{i*10}min.")
            ax.axis("off")

        # Add the colorbar
        colorbar_ax = fig.add_subplot(gs[-1, 6])
        fig.colorbar(img, cax=colorbar_ax, label="Rain [mm/h]")

        plt.tight_layout()
        
        # Save plot instead of showing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = plot_dir / f"downscaling_result_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Plot saved to: {plot_path}")
        
        plt.close(fig)
        
        return 0
    except Exception as e:
        print(f"ERROR: Failed to generate visualization: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
