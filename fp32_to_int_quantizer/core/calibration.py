import numpy as np
from typing import Tuple, Optional


class CalibrationStrategy:
    def __init__(
        self,
        quant_bit: int = 8,
        quant_level: str = "per_tensor",
        calib_mode: Optional[str] = None,
        calib_percentile: float = 99.9,
        entropy_bins: int = 2048,
        int_min: Optional[int] = None,
        int_max: Optional[int] = None,
    ):
        self.quant_bit = quant_bit
        self.quant_level = quant_level
        self.calib_mode = calib_mode
        self.calib_percentile = calib_percentile
        self.entropy_bins = entropy_bins
        self.int_min = int_min if int_min is not None else (-128 if quant_bit == 8 else -8)
        self.int_max = int_max if int_max is not None else (127 if quant_bit == 8 else 7)

    def calibrate(self, fp32_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.calib_mode is None:
            return self._calibrate_extremes(fp32_data)
        elif self.calib_mode == "percentile":
            return self._calibrate_percentile(fp32_data)
        elif self.calib_mode == "entropy":
            return self._calibrate_entropy(fp32_data)
        else:
            raise ValueError(f"Unsupported calibration mode: {self.calib_mode}")

    def _calibrate_extremes(self, fp32_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.quant_level == "per_tensor":
            return np.min(fp32_data), np.max(fp32_data)
        else:
            reduce_axes = tuple(range(1, len(fp32_data.shape)))
            calib_min = np.min(fp32_data, axis=reduce_axes)
            calib_max = np.max(fp32_data, axis=reduce_axes)
            return calib_min, calib_max

    def _calibrate_percentile(self, fp32_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.quant_level == "per_tensor":
            flat_data = fp32_data.flatten()
            calib_min = np.percentile(flat_data, 100 - self.calib_percentile)
            calib_max = np.percentile(flat_data, self.calib_percentile)
            return calib_min, calib_max
        else:
            num_channels = fp32_data.shape[1]
            calib_min = np.zeros(num_channels)
            calib_max = np.zeros(num_channels)
            for c in range(num_channels):
                channel_data = fp32_data[:, c, ...].flatten() if len(fp32_data.shape) >= 3 else fp32_data[:, c].flatten()
                calib_min[c] = np.percentile(channel_data, 100 - self.calib_percentile)
                calib_max[c] = np.percentile(channel_data, self.calib_percentile)
            return calib_min, calib_max

    def _calibrate_entropy(self, fp32_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.quant_level == "per_tensor":
            return self._entropy_calibrate_single(fp32_data)
        else:
            num_channels = fp32_data.shape[1]
            calib_min = np.zeros(num_channels)
            calib_max = np.zeros(num_channels)
            for c in range(num_channels):
                channel_data = fp32_data[:, c, ...].flatten() if len(fp32_data.shape) >= 3 else fp32_data[:, c].flatten()
                calib_min[c], calib_max[c] = self._entropy_calibrate_single(channel_data)
            return calib_min, calib_max

    def _entropy_calibrate_single(self, data: np.ndarray) -> Tuple[float, float]:
        hist, bin_edges = np.histogram(data, bins=self.entropy_bins, density=True)
        hist = hist + 1e-10

        max_entropy = -np.inf
        best_min = bin_edges[0]
        best_max = bin_edges[-1]

        min_percentile = 2 if self.quant_bit == 4 else 5
        max_percentile = 98 if self.quant_bit == 4 else 95

        for start_p in range(min_percentile, max_percentile - 1, 2):
            for end_p in range(start_p + 2, max_percentile + 1, 2):
                window_min = np.percentile(data, start_p)
                window_max = np.percentile(data, end_p)

                bin_mask = (bin_edges[:-1] >= window_min) & (bin_edges[1:] <= window_max)
                window_hist = hist[bin_mask]
                if len(window_hist) < 2:
                    continue

                window_hist_norm = window_hist / np.sum(window_hist)
                current_entropy = -np.sum(window_hist_norm * np.log2(window_hist_norm))

                if current_entropy > max_entropy:
                    max_entropy = current_entropy
                    best_min = window_min
                    best_max = window_max

        return best_min, best_max

    def compute_quant_params(
        self, calib_min: np.ndarray, calib_max: np.ndarray
    ) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
        if np.all(calib_min == calib_max):
            scales = 1.0 if isinstance(calib_min, float) else np.ones_like(calib_min, dtype=np.float32)
            zero_points = self.int_min if isinstance(calib_min, float) else np.full_like(calib_min, self.int_min, dtype=np.int32)
            return scales, zero_points

        max_abs = np.maximum(np.abs(calib_min), np.abs(calib_max))
        scales = max_abs / self.int_max
        scales[max_abs == 0] = 1.0
        zero_points = np.zeros_like(calib_min, dtype=np.int32)

        return scales, zero_points
