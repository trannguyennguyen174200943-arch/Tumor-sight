"""
Module 5: Dự đoán Tiến triển
Dự đoán sự tăng trưởng của u trong 3-6 tháng
Models: LSTM, Temporal CNN, Temporal Transformer
Sử dụng: Volume trends, morphology changes
"""
import torch  # Deep learning framework
import torch.nn as nn  # Các layer neural network
import numpy as np  # Xử lý mảng
from typing import Dict, List, Optional  # Type hints
from collections import defaultdict  # Dict mặc định


def clamp(value: float, lower: float, upper: float) -> float:
    """Giới hạn giá trị trong khoảng [lower, upper]."""
    return max(lower, min(upper, value))


def metric(morphology: Dict, *keys: str, default: float = 0.0):
    """Lấy giá trị metric từ dict morphology, dùng giá trị default nếu không tìm thấy"""
    for key in keys:  # Duyệt từng key
        if key in morphology and morphology[key] is not None:  # Nếu key tồn tại và không None
            return morphology[key]  # Trả về giá trị
    return default  # Trả về default


class ProgressionLSTM(nn.Module):
    """LSTM để dự đoán tiến triển theo thời gian"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,  # Số đặc trưng đầu vào
            hidden_size=hidden_size,  # Kích thước hidden state
            num_layers=num_layers,  # Số layer LSTM
            batch_first=True,  # Input format: (batch, seq, features)
            dropout=0.3  # Dropout để tránh overfitting
        )
        
        self.fc = nn.Sequential(
            # Fully connected layers
            nn.Linear(hidden_size, 16),  # Giảm chiều
            nn.ReLU(),  # Activation
            nn.Linear(16, 3)  # Output: 3-tháng, 6-tháng, và risk score
        )
    
    def forward(self, x):
        """Forward pass
        x shape: (batch, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)  # Chạy LSTM
        
        # Lấy hidden state cuối cùng
        last_hidden = lstm_out[:, -1, :]  # Lấy output của time step cuối
        
        # Dự đoán
        output = self.fc(last_hidden)  # Dự đoán từ hidden state
        return output


class TemporalCNN(nn.Module):
    """Temporal CNN for progression prediction"""
    
    def __init__(self, input_size: int = 10):
        super().__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Conv1d expects (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


class ProgressionAnalyzer:
    """Analyze and predict tumor progression"""
    
    def __init__(self):
        self.volume_history = defaultdict(list)
        self.morphology_history = defaultdict(list)
    
    def add_timepoint(self, case_id: str, volume: float, morphology: Dict):
        """Add a timepoint observation"""
        self.volume_history[case_id].append(volume)
        self.morphology_history[case_id].append(morphology)

    @staticmethod
    def _robust_volume_series(volumes: List[float]) -> np.ndarray:
        """Làm mượt chuỗi thể tích và giảm ảnh hưởng outlier."""
        arr = np.asarray(volumes, dtype=np.float64)
        if arr.size <= 2:
            return arr
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad <= 1e-8:
            return arr
        lower = median - (3.5 * mad)
        upper = median + (3.5 * mad)
        return np.clip(arr, lower, upper)

    @staticmethod
    def _trend_slope_per_step(values: np.ndarray) -> float:
        """Ước lượng slope bằng hồi quy tuyến tính có trọng số theo thời gian."""
        if values.size < 2:
            return 0.0
        timepoints = np.arange(values.size, dtype=np.float64)
        # Ưu tiên timepoint gần hiện tại để phản ánh diễn tiến mới nhất.
        weights = np.linspace(0.45, 1.0, values.size, dtype=np.float64)
        try:
            coeffs = np.polyfit(timepoints, values, 1, w=weights)
            return float(coeffs[0])
        except Exception:
            if values.size < 2:
                return 0.0
            return float(values[-1] - values[-2])

    @staticmethod
    def _acceleration_per_step2(values: np.ndarray) -> float:
        """Ước lượng gia tốc từ hồi quy bậc 2."""
        if values.size < 3:
            return 0.0
        timepoints = np.arange(values.size, dtype=np.float64)
        weights = np.linspace(0.45, 1.0, values.size, dtype=np.float64)
        try:
            coeffs = np.polyfit(timepoints, values, 2, w=weights)
            return float(2.0 * coeffs[0])
        except Exception:
            # Fallback bằng sai phân bậc hai tại đoạn cuối.
            return float(values[-1] - (2.0 * values[-2]) + values[-3])
    
    def predict_volume_change(self, volumes: List[float], months_ahead: int = 3) -> float:
        """
        Predict volume change using linear regression
        
        Args:
            volumes: Historical volumes
            months_ahead: Months to predict (3 or 6)
        
        Returns:
            percent_change: Predicted volume change percentage
        """
        if len(volumes) < 2:
            return 0.0
        
        volumes_arr = self._robust_volume_series(volumes)
        slope_per_step = self._trend_slope_per_step(volumes_arr)
        current_volume = float(volumes_arr[-1])
        if abs(current_volume) < 1e-8:
            return 0.0
        future_volume = current_volume + (slope_per_step * (months_ahead / 3.0))  # Assume 3-month intervals
        future_volume = max(0.0, float(future_volume))
        percent_change = ((future_volume - current_volume) / current_volume) * 100.0
        return float(np.clip(percent_change, -100.0, 300.0))
    
    def compute_growth_velocity(self, volumes: List[float]) -> float:
        """
        Compute tumor growth velocity (mm³/month)
        
        Args:
            volumes: Historical volumes in cm³
        
        Returns:
            velocity: Growth velocity in cm³/month
        """
        if len(volumes) < 2:
            return 0.0
        
        volumes_arr = self._robust_volume_series(volumes)
        slope_per_step = self._trend_slope_per_step(volumes_arr)  # cm3 / 3 months
        velocity_per_month = slope_per_step / 3.0
        return float(velocity_per_month)
    
    def compute_acceleration(self, volumes: List[float]) -> float:
        """
        Compute tumor growth acceleration (curvature)
        
        Args:
            volumes: Historical volumes
        
        Returns:
            acceleration: Growth acceleration (cm³/month²)
        """
        if len(volumes) < 3:
            return 0.0
        
        volumes_arr = self._robust_volume_series(volumes)
        acceleration_per_step2 = self._acceleration_per_step2(volumes_arr)  # cm3 / (3 months)^2
        acceleration_per_month2 = acceleration_per_step2 / 9.0
        return float(acceleration_per_month2)
    
    def predict_malignant_transition(self, morphology_history: List[Dict]) -> float:
        """
        Predict risk of malignant transition
        Based on morphology changes
        
        Args:
            morphology_history: List of morphology dicts over time
        
        Returns:
            risk: Risk probability (0-1)
        """
        if len(morphology_history) < 2:
            return 0.0

        def _trend_norm(values: List[float], scale: float) -> float:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size < 2:
                return 0.0
            slope = self._trend_slope_per_step(arr)
            return clamp(slope / max(scale, 1e-8), -1.0, 1.0)

        irregularity_trend = _trend_norm(
            [float(metric(m, "surface_irregularity", "surface_irregularity_index", default=0.0)) for m in morphology_history],
            0.12,
        )
        elongation_trend = _trend_norm([float(m.get("elongation", 1.0)) for m in morphology_history], 0.35)
        heterogeneity_trend = _trend_norm(
            [float(metric(m, "std_intensity", "intensity_std", default=0.0)) for m in morphology_history],
            10.0,
        )
        fractal_trend = _trend_norm([float(m.get("fractal_dimension", 2.0)) for m in morphology_history], 0.14)

        latest = morphology_history[-1]
        latest_irregularity = float(metric(latest, "surface_irregularity", "surface_irregularity_index", default=0.0))
        latest_heterogeneity = float(metric(latest, "std_intensity", "intensity_std", default=0.0))
        latest_fractal = float(latest.get("fractal_dimension", 2.0))

        base_risk = (
            clamp((latest_irregularity - 0.18) / 0.42, 0.0, 1.0) * 0.22
            + clamp((latest_heterogeneity - 8.0) / 28.0, 0.0, 1.0) * 0.18
            + clamp((latest_fractal - 2.05) / 0.5, 0.0, 1.0) * 0.12
        )
        trend_risk = (
            max(0.0, irregularity_trend) * 0.2
            + max(0.0, elongation_trend) * 0.12
            + max(0.0, heterogeneity_trend) * 0.11
            + max(0.0, fractal_trend) * 0.05
        )
        stability_bonus = (
            max(0.0, -irregularity_trend) * 0.06
            + max(0.0, -heterogeneity_trend) * 0.04
        )

        risk = clamp(base_risk + trend_risk - stability_bonus, 0.0, 1.0)
        return float(risk)
    
    def predict_progression(self, case_id: str, months_ahead: int = 3) -> Dict:
        """
        Complete progression prediction
        
        Args:
            case_id: Case identifier
            months_ahead: Months to predict
        
        Returns:
            prediction: Dict with predictions
        """
        volumes = self.volume_history.get(case_id, [])
        morphologies = self.morphology_history.get(case_id, [])
        
        if len(volumes) < 2:
            return {
                "volume_change_percent": 0.0,
                "growth_velocity": 0.0,
                "growth_acceleration": 0.0,
                "malignant_risk": 0.0,
                "summary": "Chưa đủ dữ liệu để dự báo"
            }
        
        volume_change = self.predict_volume_change(volumes, months_ahead)
        velocity = self.compute_growth_velocity(volumes)
        acceleration = self.compute_acceleration(volumes)
        malignant_risk = self.predict_malignant_transition(morphologies)
        
        # Generate summary
        if volume_change > 30:
            summary = f"Khối u tăng nhanh (+{volume_change:.1f}% trong {months_ahead} tháng), nguy cơ cao"
        elif volume_change > 10:
            summary = f"Khối u tăng mức vừa (+{volume_change:.1f}%), cần theo dõi sát"
        elif volume_change < -10:
            summary = f"Khối u giảm kích thước ({volume_change:.1f}%), có thể đang đáp ứng điều trị"
        else:
            summary = f"Khối u tương đối ổn định (~{volume_change:.1f}% thay đổi)"
        
        return {
            "volume_change_percent": volume_change,
            "growth_velocity": velocity,
            "growth_acceleration": acceleration,
            "malignant_risk": malignant_risk,
            "summary": summary
        }


if __name__ == "__main__":
    # Test
    analyzer = ProgressionAnalyzer()
    
    # Simulate volume measurements over time
    volumes = [30.0, 35.5, 42.3, 51.2]  # Growing tumor
    
    pred = analyzer.predict_volume_change(volumes, months_ahead=3)
    print(f"Volume change in 3 months: {pred:.1f}%")
    
    pred6 = analyzer.predict_volume_change(volumes, months_ahead=6)
    print(f"Volume change in 6 months: {pred6:.1f}%")
    
    velocity = analyzer.compute_growth_velocity(volumes)
    print(f"Growth velocity: {velocity:.2f} cm³/month")
