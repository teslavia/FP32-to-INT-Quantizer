import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Dict


def evaluate_precision(
    fp32_original: np.ndarray,
    fp32_recovered: np.ndarray,
    quant_bit: int = 8,
    sig_analysis: bool = True,
) -> Dict:
    error = fp32_original - fp32_recovered
    squared_error = error ** 2
    absolute_error = np.abs(error)

    metrics = {
        "quant_bit": quant_bit,
        "MSE": np.mean(squared_error),
        "MAE": np.mean(absolute_error),
        "RMSE": np.sqrt(np.mean(squared_error)),
        "PSNR": 10 * np.log10((np.max(fp32_original) ** 2) / np.mean(squared_error)) if np.mean(squared_error) != 0 else np.inf,
        "SNR": 10 * np.log10(np.var(fp32_original) / np.var(error)) if np.var(error) != 0 else np.inf,
        "Max_Abs_Error": np.max(absolute_error),
        "Min_Abs_Error": np.min(absolute_error),
        "95th_Percentile_Error": np.percentile(absolute_error, 95),
    }

    if sig_analysis:
        metrics.update({
            "Error_Std": np.std(error),
            "Error_Var": np.var(error),
            "Error_Skewness": np.mean(((error - np.mean(error)) / np.std(error)) ** 3) if np.std(error) != 0 else 0.0,
            "Error_Kurtosis": np.mean(((error - np.mean(error)) / np.std(error)) ** 4) - 3 if np.std(error) != 0 else 0.0,
            "Error_Entropy": scipy_entropy(np.histogram(error, bins=100, density=True)[0] + 1e-10),
        })

    return metrics
