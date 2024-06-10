import numpy as np
from pathlib import Path
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def isnormalized(data: np.ndarray) -> bool:
    return data.mean(axis=0) == 0


def shift_signal(signal1: np.ndarray, signal2: np.ndarray, optimal_lag: int) -> np.ndarray:
    
    if optimal_lag > 0:
        shifted_signal2 = np.pad(signal2, (optimal_lag, 0), 'constant', constant_values=(0, 0))[:len(signal1)]
    elif optimal_lag < 0:
        shifted_signal2 = np.pad(signal2, (0, -optimal_lag), 'constant', constant_values=(0, 0))[-len(signal1):]
    else:
        shifted_signal2 = signal2
    
    min_length = min(len(signal1), len(shifted_signal2))
    signal1 = signal1[:min_length]
    shifted_signal2 = shifted_signal2[:min_length]

    return shifted_signal2


class Statistics:
    def __init__(
        self,
        statistics_config_list: list[str],
    ) -> None:
        
        self.statistics_list = statistics_config_list
        self.statistics = {
            "cross_correlation": self.cross_correlation, 
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "dtw": self.dtw,
            "cosine": self.cosine_sim
        }
    
    def __call__(
        self, 
        real_data: np.ndarray, 
        prediction: np.ndarray
    ) -> dict[str, float]:
        
        if not isnormalized(real_data):
            real_data = (real_data - real_data.mean(axis=0)) / (real_data.std(axis=0) + 1e-6)
        
        if not isnormalized(prediction):
            prediction = (prediction - prediction.mean(axis=0)) / (prediction.std(axis=0) / 1e-6)
        
        if len(real_data) != prediction:
            real_data = signal.resample(real_data, prediction.shape[0], axis=0)
        
        # Transform the data to be aligned with the image-space predictions
        real_data = real_data[:, [2, 1, 0]]
        real_data = real_data[:, :2]

        statistics = {}
        
        if "cross_correlation" in self.statistics_list:
            idx = self.statistics_list.index("cross_correlation")
            statistics["cross_correlation"], optimal_lag = self.cross_correlation(real_data, prediction)
            self.statistics_list.pop(idx)
        
        else:
            optimal_lag = 0

        for stat in self.statistics_list:
            statistics[stat] = self.statistics[stat](real_data, prediction, optimal_lag)

        return statistics
    
    @staticmethod
    def rmse(
        real_data: np.ndarray,
        prediction: np.ndarray,
        optimal_lag: int = 0
    ) -> float:
        
       shifted_pred = shift_signal(real_data, prediction, optimal_lag)
       errors = real_data - shifted_pred
       squared_errors = errors ** 2
       mean_squared_error = np.mean(squared_errors)
       rmse = np.sqrt(mean_squared_error)
       return rmse
    
    @staticmethod
    def mae(
        real_data: np.ndarray,
        prediction: np.ndarray,
        optimal_lag: int = 0
    ) -> dict[str, float]:
        
        shifted_pred = shift_signal(real_data, prediction, optimal_lag)
        abs_errors = np.abs(real_data - shifted_pred)
        mae = np.mean(abs_errors)
        return mae   

    @staticmethod
    def cross_correlation(
        real_data: np.ndarray, 
        prediction: np.ndarray
    ) -> dict[str, float]:
        
        # Compute the cross-correlation
        corr = np.correlate(prediction[:, 0], real_data[:, 0], mode="full")
        normalized_corr = corr / (np.sqrt(np.sum(prediction[:, 0] ** 2) * np.sum(real_data[:, 0] ** 2)) + 1e-6)

        lags = np.arange(-real_data.shape[0] + 1, real_data.shape[0])

        # Find the lag with maximum correlation
        max_corr_index = np.argmax(normalized_corr)
        max_corr = normalized_corr[max_corr_index]
        optimal_lag = lags[max_corr_index]
        return max_corr, optimal_lag
    
    @staticmethod
    def mape(
        real_data: np.ndarray,
        prediction: np.ndarray,
        optimal_lag: int = 0
    ) -> dict[str, float]:
        
        shifted_pred = shift_signal(real_data, prediction, optimal_lag)
        frac_error = np.abs((real_data - shifted_pred) / real_data)
        mape = np.mean(frac_error) * 100
        return mape
    
    @staticmethod
    def dtw(
        real_data: np.ndarray,
        prediction: np.ndarray,
        optimal_lag: int = 0
    ) -> dict[str, float]:
        
        shifted_pred = shift_signal(real_data, prediction, optimal_lag)
        distance, path = fastdtw(real_data, shifted_pred, dist=euclidean)
        return distance
    

    @staticmethod
    def cosine_sim(
        real_data: np.ndarray,
        prediction: np.ndarray,
        optimal_lag: int = 0
    ) -> dict[str, float]:
        
        shifted_pred = shift_signal(real_data, prediction, optimal_lag)
        dot_product = np.dot(real_data, shifted_pred)

        # Calculate the L2 norms of the data
        norm_real_data = np.linalg.norm(real_data)
        norm_pred = np.linalg.norm(shifted_pred)

        # Calculate the cosine similarity
        cosine_similarity = dot_product / (norm_real_data * norm_pred)
        return cosine_similarity

