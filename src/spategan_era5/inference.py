import torch
import numpy as np
from typing import Optional, Union


class InferenceEngine:
    def __init__(self, model: torch.nn.Module, sliding_step: int = 1, device: Optional[torch.device] = None):
        self.model = model
        self.sliding_step = sliding_step
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).float()
        return array.to(self.device)

    def infer(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target: Optional[Union[np.ndarray, torch.Tensor]] = None,
        seed: int = 1,
        return_numpy: bool = True
    ) -> Union[tuple, np.ndarray, torch.Tensor]:
        """
        Run inference with sliding windows over time.

        Args:
            x: Input of shape (B, C, T, H, W) as torch.Tensor or np.ndarray
            target: Ground truth target (optional)
            return_numpy: Whether to return predictions as numpy (True) or torch tensor (False)

        Returns:
            Predictions as numpy or torch.Tensor
        """
        x = self._to_tensor(x)

        ## move to separete data processing script
        # Set values < 0 to 0
        x = torch.clamp(x, min=0.0)
    
        # Assert no NaNs in the input
        if torch.isnan(x).any():
            raise ValueError("Input contains NaNs. Please check the input data.")
        
        predictions = []
        T = x.shape[2]

        for i in range(4, T - 4, self.sliding_step):
            if i + 12 > T:
                continue

            first_slice = -1 * ((self.sliding_step * 6) - 48) // 2
            last_slice = ((self.sliding_step * 6) - 48) // 2

            if i - 4 == 0:
                first_slice = 0
            if i + 12 == T:
                last_slice = 48

            x_window = x[:, :, i - 4: i + 12]

            with torch.no_grad():
                pred = self.model(x_window, seed).cpu()

            pred = pred[:, :, first_slice:last_slice]
            predictions.append(pred)

        # Combine predictions
        predictions = torch.cat(predictions, dim=2)

        # Apply cropping to target and predictions to remove extended ERA5 informatoin (+-4hr) and boundary areas
        if target is not None:
            target = self._to_tensor(target)
            target = target[0, 0, 24:-24, 12:-12, 12:-12]
            target = target[6:-6].cpu()
            predictions = predictions[0, 0, 6:-6, 12:-12, 12:-12]
        else:
            predictions = predictions[0, 0]  # remove batch,channel

        # Input data to target domain
        x = x.cpu()[0,:,5:-5, 8:-8, 8:-8]
        x = x.sum(dim=0)
        
       # Return predictions and target in mm/h
        if return_numpy:
            return predictions.numpy()*6, target.numpy()*6, x.numpy() if target is not None else predictions.numpy()*6
        else:
            return predictions*6, target*6, x if target is not None else predictions*6