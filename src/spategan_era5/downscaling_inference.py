"""
Inference wrapper module for flexible ERA5 downscaling.

Handles model loading, data preparation, and inference execution.
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from src.spategan_era5.model import Generator


class ERA5DownscalingInference:
    """Wrapper for spateGAN model inference on ERA5 data.
    
    Handles model loading, tensor preparation, and sliding window prediction
    for high-resolution precipitation downscaling.
    
    Args:
        config: Configuration dictionary with model and processing settings.
        device: Compute device ('cuda' or 'cpu').
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        config: dict,
        device: str = 'cuda',
        seed: int = 42,
    ) -> None:
        """
        Initialize the inference engine.
        
        Args:
            model_weights_path: Path to model weights file
            device: 'cuda' or 'cpu'
            seed: Random seed for reproducibility
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        model_weights_path = config['data']['model_weights_path']
        self.final_era_constraint = config['inference'].get('final_era_constraint', True)        
        
        # Load model
        self.model = Generator().to(self.device)
        checkpoint = torch.load(model_weights_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
    
    def prepare_tensor_dataset(
        self,
        dataset: xr.Dataset,
        variable_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Prepare tensor dataset from xarray Dataset.
        
        Args:
            dataset: Input xarray Dataset containing variables
            variable_names: List of variable names to extract. If None, uses all data variables
        
        Returns:
            torch.Tensor of shape (1, channels, time, height, width)
        
        Raises:
            ValueError: If requested variables not found in dataset
        """
        if variable_names is None:
            variable_names = list(dataset.data_vars)
        
        data_arrays: list[np.ndarray] = []
        
        for var_name in variable_names:
            if var_name not in dataset.data_vars:
                raise ValueError(f"Variable '{var_name}' not found in dataset")
            
            data = dataset[var_name].values  # (time, y, x)
            data_arrays.append(data)
        
        # Stack variables: list of (time, y, x) -> (channels, time, y, x)
        stacked_data = np.stack(data_arrays, axis=0)
        
        # Add batch dimension: (channels, time, y, x) -> (1, channels, time, y, x)
        batched_data = np.expand_dims(stacked_data, axis=0)
        
        # Convert to tensor and move to device
        tensor = torch.from_numpy(batched_data).float().to(self.device)
        
        return tensor
    
    def predict_sliding_window(
        self,
        x: torch.Tensor,
        ds_prediction: xr.Dataset | None = None,
        slide: int = 8,
    ) -> np.ndarray | xr.Dataset:
        """
        Apply model with sliding window and concatenate predictions.
        
        Args:
            x: Input tensor of shape (batch, channels, time, height, width) with time >= 16
            slide: Sliding window step size in hours (1-8)
        
        Returns:
            numpy.ndarray of shape (time * 6, height, width) containing concatenated predictions.
                    The 6 comes from 10-minute resolution (6 steps per hour).
        
        Raises:
            ValueError: If input has fewer than 16 timesteps or slide is out of range
        """
        n_times = x.shape[2]
        h, w = 144, 144
        
        if n_times < 16:
            raise ValueError(f"Input must have at least 16 timesteps, got {n_times}")
        
        if not 1 <= slide <= 8:
            raise ValueError(f"Slide must be between 1 and 8, got {slide}")
        
        steps_per_hour = 6  # 10-minute resolution
        steps_to_keep = slide * steps_per_hour
        
        # Calculate center indices for middle windows
        center_start = (48 - steps_to_keep) // 2
        center_end = center_start + steps_to_keep
        
        # Initialize predictions list with NaN padding for first 4 hours
        predictions: list[np.ndarray] = [
            np.full((4 * steps_per_hour, h, w), np.nan)
        ]
                
        # Generate sliding window positions
        positions = list(range(0, n_times - 15, slide))
        
        for i, pos in enumerate(tqdm(positions, desc='Downscaling process', dynamic_ncols=True)):
            x_window = x[:, :, pos:pos + 16]
            
            with torch.no_grad():
                if self.seed == -1:
                    seed = torch.randint(0, 10000, (1,))
                else:
                    seed = self.seed
                pred = self.model(x_window, seed)  # Shape: (batch, channels, 48, h, w)
                
            
            # Extract predictions, remove batch and channel dims, crop boundaries
            pred = pred.detach().cpu().numpy()[0, 0, :, 12:-12, 12:-12]
            
            is_first = (i == 0)
            is_last = (i == len(positions) - 1)
            
            if is_first and is_last:
                # Only one window: keep everything
                predictions.append(pred)
            elif is_first:
                # First window: keep from start to center_end
                predictions.append(pred[:center_end])
            elif is_last:
                # Last window: keep from center_start to end
                predictions.append(pred[center_start:])
            else:
                # Middle windows: keep only center portion
                predictions.append(pred[center_start:center_end])
        
        # Concatenate all predictions
        result = np.concatenate(predictions, axis=0)
        
        # Pad with NaNs if necessary to match expected output length
        expected_length = n_times * steps_per_hour
        current_length = result.shape[0]
        
        if current_length < expected_length:
            end_padding = np.full((expected_length - current_length, h, w), np.nan)
            result = np.concatenate([result, end_padding], axis=0)
            
            
        if self.final_era_constraint:
            # Apply final ERA5 constraint
            constraint = x[0,:, 4:-4, 8:-8, 8:-8].sum(dim=0, keepdim=False).detach().cpu().numpy()
            scale = constraint.mean() / 6
            pred_mean = np.nanmean(result)
            result = result * (scale / pred_mean)
            
            
        if ds_prediction is not None:
            if result.shape[0] != len(ds_prediction.time):
                raise ValueError(f"Prediction length {result.shape[0]} does not match ds_prediction time length {len(ds_prediction.time)}")
            else:
                ds_output = ds_prediction.copy(deep=True)
                # Keep spatial dims consistent with UTM datasets (y northing, x easting)
                ds_output['precipitation'] = (['time', 'y', 'x'], result*6)
                ds_output['precipitation'].attrs['units'] = 'mm/h'
                ds_output.attrs['time zone'] = 'UTC'
                return ds_output # return precipitation prediction in mm/h
        else:
            return result * 6 # return precipitation prediction in mm/h
    