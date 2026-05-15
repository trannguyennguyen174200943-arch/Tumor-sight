"""
Module 4: Phân loại (Lành tính vs Ác tính)
Sử dụng các đặc trưng hình thái và radiomics để phân loại u
Models: 3D CNN, ResNet 3D, XGBoost
"""
import json
import logging  # Ghi log
import os
from pathlib import Path  # Xử lý đường dẫn
from typing import Any, Dict, Optional, Tuple  # Type hints

import numpy as np  # Xử lý mảng
import torch  # Deep learning framework
import torch.nn as nn  # Các layer neural network
import torch.nn.functional as F  # Các hàm functional
from sklearn.ensemble import GradientBoostingClassifier  # XGBoost classifier
import pickle  # Lưu/tải object Python


logger = logging.getLogger(__name__)  # Logger cho module

DEFAULT_CLASSIFICATION_TUNING_PATH = Path(
    os.getenv(
        "CLASSIFICATION_TUNING_PATH",
        str(Path(__file__).resolve().parents[1] / "checkpoints" / "benchmark" / "classification_tuning.json"),
    )
)


def metric(morphology: Dict, *keys: str, default: float = 0.0):
    """Lấy giá trị metric từ dict morphology, dùng giá trị default nếu không tìm thấy"""
    for key in keys:  # Duyệt từng key
        if key in morphology and morphology[key] is not None:  # Nếu key tồn tại và không None
            return morphology[key]  # Trả về giá trị
    return default  # Trả về default


def clamp(value: float, lower: float, upper: float) -> float:
    """Giới hạn giá trị trong khoảng [lower, upper]"""
    return max(lower, min(upper, value))  # Đảm bảo value nằm trong khoảng


def normalize(value: float, lower: float, upper: float) -> float:
    """Chuẩn hóa giá trị về khoảng [0, 1]"""
    if upper <= lower:  # Nếu khoảng không hợp lệ
        return 0.0  # Trả về 0
    return clamp((value - lower) / (upper - lower), 0.0, 1.0)  # Chuẩn hóa và giới hạn


class ClassificationCNN(nn.Module):
    """3D CNN đơn giản để phân loại u"""
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),  # Convolution 3x3x3
            nn.BatchNorm3d(32),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool3d(2),  # Max pooling 2x2x2
            
            # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv3d(32, 64, kernel_size=3, padding=1),  # Tăng số filter
            nn.BatchNorm3d(64),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool3d(2),  # Max pooling
            
            # Block 3: Conv -> BatchNorm -> ReLU -> GlobalAvgPool
            nn.Conv3d(64, 128, kernel_size=3, padding=1),  # Tiếp tục tăng filter
            nn.BatchNorm3d(128),  # Batch normalization
            nn.ReLU(inplace=True),  # Activation function
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
        )
        
        self.classifier = nn.Sequential(
            # Fully connected layers
            nn.Linear(128, 64),  # Hidden layer
            nn.ReLU(inplace=True),  # Activation
            nn.Dropout(0.5),  # Dropout để tránh overfitting
            nn.Linear(64, 2)  # Output: 2 classes (benign/malignant)
        )
    
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)  # Trích xuất features
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # Phân loại
        return x


class FeatureBasedClassifier:
    """
    Phân loại u bằng cách sử dụng morphology và radiomics features
    Sử dụng XGBoost hoặc scoring dựa trên luật
    """
    
    # Rule-based scoring thresholds
    BENIGN_FEATURES = {
        "sphericity_high": lambda m: m.get("sphericity", 0) > 0.7,
        "smooth_surface": lambda m: metric(m, "surface_irregularity", "surface_irregularity_index", default=1) < 0.2,
        "homogeneous_intensity": lambda m: metric(m, "std_intensity", "intensity_std", default=100) < 10,
        "compact_shape": lambda m: m.get("compactness", 0) > 0.5,
    }
    
    MALIGNANT_FEATURES = {
        "irregular_surface": lambda m: metric(m, "surface_irregularity", "surface_irregularity_index", default=0) > 0.4,
        "elongated_shape": lambda m: m.get("elongation", 1) > 1.8,
        "heterogeneous_intensity": lambda m: metric(m, "std_intensity", "intensity_std", default=0) > 25,
        "high_grad_energy": lambda m: metric(m, "grad_energy", "gradient_energy", default=0) > 150,
        "fractal_boundary": lambda m: m.get("fractal_dimension", 2) > 2.3,
    }
    
    def __init__(self):
        self.xgb_model = None

    def _weighted_components(self, morphology: Dict) -> Dict[str, Tuple[float, str]]:
        irregularity = float(metric(morphology, "surface_irregularity", "surface_irregularity_index", default=0.0))
        elongation = float(morphology.get("elongation", 1.0))
        intensity_std = float(metric(morphology, "std_intensity", "intensity_std", "intensity_heterogeneity_index", default=0.0))
        gradient_energy = float(metric(morphology, "grad_energy", "gradient_energy", "gradient_distribution", default=0.0))
        fractal = float(morphology.get("fractal_dimension", 2.0))
        volume_cm3 = float(metric(morphology, "volume_cm3", "tumor_volume_cm3", default=0.0))
        sphericity = float(morphology.get("sphericity", 0.0))
        compactness = float(morphology.get("compactness", 0.0))
        margin_complexity = float(metric(morphology, "margin_complexity_index", default=0.0))

        return {
            "bo_khong_deu": (
                normalize(irregularity, 0.18, 0.58) * 0.22,
                "Bờ tổn thương không đều",
            ),
            "phuc_tap_duong_bien": (
                normalize(fractal, 2.05, 2.55) * 0.17,
                "Đường biên có độ phức tạp cao",
            ),
            "khong_dong_nhat_cuong_do": (
                normalize(intensity_std, 8.0, 42.0) * 0.14,
                "Tín hiệu trong khối không đồng nhất",
            ),
            "keo_dai_hinh_dang": (
                normalize(elongation, 1.15, 2.35) * 0.11,
                "Hình dạng kéo dài theo một trục",
            ),
            "the_tich_lon": (
                normalize(volume_cm3, 4.0, 55.0) * 0.1,
                "Thể tích tổn thương lớn",
            ),
            "gradient_manh": (
                normalize(gradient_energy, 10.0, 120.0) * 0.08,
                "Biến thiên cường độ theo biên rõ",
            ),
            "giam_do_tron": (
                normalize(1.0 - sphericity, 0.1, 0.7) * 0.09,
                "Hình khối giảm độ tròn đều",
            ),
            "giam_do_dac": (
                normalize(0.55 - compactness, 0.0, 0.45) * 0.05,
                "Mức độ đặc hình học thấp",
            ),
            "margin_complexity": (
                normalize(margin_complexity, 0.18, 0.85) * 0.04,
                "Độ phức tạp bờ cao",
            ),
        }
    
    def rule_based_classify(self, morphology: Dict) -> Tuple[str, float]:
        """
        Rule-based classification using morphology features
        
        Args:
            morphology: Dict of morphology features
        
        Returns:
            label: "benign" or "malignant"
            confidence: Mức tin cậy (0-1)
        """
        risk_score = self.compute_risk_score(morphology)
        if risk_score >= 65:
            return "malignant", clamp(0.55 + ((risk_score - 65) / 35.0) * 0.4, 0.55, 0.95)
        if risk_score <= 35:
            return "benign", clamp(0.55 + ((35 - risk_score) / 35.0) * 0.35, 0.55, 0.9)
        return "uncertain", clamp(0.45 + abs(risk_score - 50.0) / 50.0 * 0.12, 0.45, 0.6)
    
    def compute_risk_score(self, morphology: Dict) -> float:
        """
        Compute malignancy risk score (0-100)
        
        Args:
            morphology: Dict of morphology features
        
        Returns:
            risk_score: Score from 0 (benign) to 100 (malignant)
        """
        components = self._weighted_components(morphology)
        risk_score = sum(weight for weight, _ in components.values()) * 100.0

        if self.BENIGN_FEATURES["sphericity_high"](morphology):
            risk_score -= 8.0
        if self.BENIGN_FEATURES["smooth_surface"](morphology):
            risk_score -= 10.0
        if self.BENIGN_FEATURES["compact_shape"](morphology):
            risk_score -= 6.0

        return float(clamp(risk_score, 0.0, 100.0))


class ClassificationEngine:
    """Main classification engine with optional 3D checkpoint support."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.feature_classifier = FeatureBasedClassifier()
        self.cnn_model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = 2
        self.target_size = (64, 64, 64)
        self.model_name = "Radiomics + morphology"
        self.runtime_mode = "feature_based"
        self.tuning = self._load_tuning_payload()
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self._load_checkpoint()

    def _load_tuning_payload(self) -> Dict[str, Any]:
        path = DEFAULT_CLASSIFICATION_TUNING_PATH
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            payload["_path"] = str(path)
            return payload
        except Exception as exc:
            logger.warning("Failed to load classification tuning payload %s: %s", path, exc)
            return {}

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[Path]) -> Optional[Path]:
        if checkpoint_path:
            path = Path(checkpoint_path)
            return path if path.exists() else None

        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "checkpoints" / "classification_3d" / "best.pt",
            repo_root / "checkpoints" / "resnet3d_classifier" / "best.pt",
            repo_root / "checkpoints" / "tumor_classifier_3d" / "best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(payload, dict):
                state_dict = payload.get("model_state_dict") or payload.get("state_dict") or payload
                config = payload.get("config", {})
                self.input_channels = int(payload.get("input_channels", config.get("input_channels", self.input_channels)))
                self.model_name = str(payload.get("model_name", "3D CNN classifier"))
            else:
                state_dict = payload

            model = ClassificationCNN(input_channels=self.input_channels)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.cnn_model = model
            self.runtime_mode = "checkpoint_3d"
        except Exception as exc:
            logger.warning("Failed to load 3D classifier checkpoint %s: %s", self.checkpoint_path, exc)
            self.cnn_model = None
            self.runtime_mode = "feature_based"
            self.model_name = "Radiomics + morphology"

    @staticmethod
    def _normalize_volume(volume: np.ndarray) -> np.ndarray:
        image = np.asarray(volume, dtype=np.float32)
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros_like(image, dtype=np.float32)
        image = np.where(finite, image, 0.0)
        sample = image[np.abs(image) > 1e-6]
        if sample.size == 0:
            sample = image[finite]
        lower, upper = np.percentile(sample, [1.0, 99.0])
        image = np.clip(image, lower, upper)
        mean = float(sample.mean()) if sample.size else float(image.mean())
        std = float(sample.std()) if sample.size else float(image.std())
        return ((image - mean) / max(std, 1e-6)).astype(np.float32)

    def _crop_to_roi(self, channels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return channels

        lower = np.maximum(coords.min(axis=0) - 6, 0)
        upper = np.minimum(coords.max(axis=0) + 7, np.asarray(mask.shape))
        z0, y0, x0 = [int(item) for item in lower]
        z1, y1, x1 = [int(item) for item in upper]
        return channels[:, z0:z1, y0:y1, x0:x1]

    def _prepare_checkpoint_input(self, volume: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        image = np.asarray(volume, dtype=np.float32)
        if image.ndim == 4:
            normalized = [self._normalize_volume(image[index]) for index in range(min(image.shape[0], max(1, self.input_channels - 1)))]
            if not normalized:
                normalized = [self._normalize_volume(image.mean(axis=0))]
        else:
            normalized = [self._normalize_volume(image)]

        if self.input_channels <= 1:
            channels = np.stack([normalized[0]], axis=0).astype(np.float32)
        else:
            while len(normalized) < self.input_channels - 1:
                normalized.append(normalized[-1])
            channels = np.stack(normalized[: self.input_channels - 1] + [mask.astype(np.float32)], axis=0).astype(np.float32)

        cropped = self._crop_to_roi(channels, mask)
        tensor = torch.from_numpy(cropped[None]).to(self.device)
        tensor = F.interpolate(tensor, size=self.target_size, mode="trilinear", align_corners=False)
        return tensor

    def _checkpoint_probability(self, volume: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if self.cnn_model is None:
            return None

        try:
            with torch.no_grad():
                tensor = self._prepare_checkpoint_input(volume, mask)
                logits = self.cnn_model(tensor)
                probability = torch.softmax(logits, dim=1)[0, 1].item()
            return float(np.clip(probability, 0.0, 1.0))
        except Exception as exc:
            logger.warning("3D checkpoint classification failed: %s", exc)
            return None

    def _calibrate_probability(self, probability: float) -> float:
        """Hiệu chỉnh xác suất theo temperature scaling để giảm over-confident."""
        temperature = float(self.tuning.get("temperature", 1.0) or 1.0)
        temperature = max(0.25, min(3.0, temperature))
        clipped = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        logit = np.log(clipped / (1.0 - clipped))
        calibrated = 1.0 / (1.0 + np.exp(-logit / temperature))
        return float(np.clip(calibrated, 0.0, 1.0))

    def _decision_thresholds(self) -> Tuple[float, float]:
        benign_threshold = float(self.tuning.get("benign_threshold", 0.35) or 0.35)
        malignant_threshold = float(self.tuning.get("malignant_threshold", 0.65) or 0.65)
        benign_threshold = float(np.clip(benign_threshold, 0.15, 0.5))
        malignant_threshold = float(np.clip(malignant_threshold, 0.5, 0.85))
        if benign_threshold >= malignant_threshold:
            benign_threshold, malignant_threshold = 0.35, 0.65
        return benign_threshold, malignant_threshold

    def classify(self, morphology: Dict, volume: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> Dict:
        feature_label, feature_confidence = self.feature_classifier.rule_based_classify(morphology)
        feature_risk_score = self.feature_classifier.compute_risk_score(morphology)
        reasoning = self._explain_classification(morphology)

        checkpoint_probability = None
        if volume is not None and mask is not None:
            checkpoint_probability = self._checkpoint_probability(volume, mask)

        benign_threshold, malignant_threshold = self._decision_thresholds()
        if checkpoint_probability is None:
            feature_probability = self._calibrate_probability(feature_risk_score / 100.0)
            if feature_probability >= malignant_threshold:
                label = "malignant"
            elif feature_probability <= benign_threshold:
                label = "benign"
            else:
                label = "uncertain"
            confidence = clamp(0.52 + abs(feature_probability - 0.5) * 0.75, 0.5, 0.9)
            return {
                "label": label or feature_label,
                "confidence": float(confidence if label != "uncertain" else min(confidence, feature_confidence)),
                "risk_score": float(np.clip(feature_probability * 100.0, 0.0, 100.0)),
                "reasoning": reasoning,
            }

        feature_weight = float(self.tuning.get("feature_weight", 0.45) or 0.45)
        checkpoint_weight = float(self.tuning.get("checkpoint_weight", 0.55) or 0.55)
        feature_weight = float(np.clip(feature_weight, 0.05, 0.95))
        checkpoint_weight = float(np.clip(checkpoint_weight, 0.05, 0.95))
        norm = max(feature_weight + checkpoint_weight, 1e-6)
        feature_weight /= norm
        checkpoint_weight /= norm

        feature_probability = self._calibrate_probability(feature_risk_score / 100.0)
        checkpoint_probability = self._calibrate_probability(checkpoint_probability)
        blended_probability = clamp((feature_probability * feature_weight) + (checkpoint_probability * checkpoint_weight), 0.0, 1.0)

        if blended_probability >= malignant_threshold:
            label = "malignant"
        elif blended_probability <= benign_threshold:
            label = "benign"
        else:
            label = "uncertain"

        agreement = 1.0 - abs(feature_probability - checkpoint_probability)
        confidence = clamp(
            0.52 + abs(blended_probability - 0.5) * 0.72 + agreement * 0.12,
            0.55,
            0.98,
        )
        if label == "uncertain":
            confidence = min(confidence, 0.72)
        risk_score = float(np.clip(blended_probability * 100.0, 0.0, 100.0))
        checkpoint_reason = f"Checkpoint 3D ước lượng xác suất ác tính {checkpoint_probability * 100.0:.1f}%"
        calibration_reason = (
            f"Hiệu chỉnh ngưỡng benign/malignant {benign_threshold:.2f}/{malignant_threshold:.2f}"
            f", blend feature/checkpoint {feature_weight:.2f}/{checkpoint_weight:.2f}"
        )
        if self.tuning.get("_path"):
            calibration_reason += " (đã tune)"
        return {
            "label": label,
            "confidence": confidence,
            "risk_score": risk_score,
            "reasoning": f"{reasoning} | {checkpoint_reason} | {calibration_reason}",
        }

    def _explain_classification(self, morphology: Dict) -> str:
        """Generate explanation for classification"""
        ranked = sorted(
            self.feature_classifier._weighted_components(morphology).values(),
            key=lambda item: item[0],
            reverse=True,
        )
        explanations = [description for weight, description in ranked if weight >= 0.08][:3]

        if morphology.get("sphericity", 0) > 0.72 and metric(morphology, "surface_irregularity", "surface_irregularity_index", default=1) < 0.2:
            explanations.append("Hình thái tròn đều làm giảm khả năng ác tính")
        if morphology.get("compactness", 0) > 0.55:
            explanations.append("Cấu trúc đặc tương đối ủng hộ tính lành")

        return " | ".join(explanations[:4]) if explanations else "Đánh giá hình thái chuẩn"


if __name__ == "__main__":
    # Test
    engine = ClassificationEngine()
    
    # Benign example
    benign_morph = {
        "surface_irregularity": 0.15,
        "sphericity": 0.85,
        "elongation": 1.1,
        "std_intensity": 8,
        "compactness": 0.7,
        "fractal_dimension": 2.0,
        "grad_energy": 50,
    }
    
    # Malignant example
    malignant_morph = {
        "surface_irregularity": 0.5,
        "sphericity": 0.4,
        "elongation": 2.5,
        "std_intensity": 40,
        "compactness": 0.2,
        "fractal_dimension": 2.6,
        "grad_energy": 200,
    }
    
    print("Benign Case:")
    print(engine.classify(benign_morph))
    print("\nMalignant Case:")
    print(engine.classify(malignant_morph))
