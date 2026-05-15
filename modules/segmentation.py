"""Các engine phân đoạn được sử dụng bởi backend TumorSight."""

from __future__ import annotations

import json  # Để xử lý JSON
import os  # Để truy cập biến môi trường
import pickle  # Để lưu/tải object Python
import sys
import threading  # Để thực thi đa luồng
from dataclasses import dataclass, field  # Để tạo class dữ liệu
from pathlib import Path  # Để xử lý đường dẫn
from typing import Any, Dict, Optional, Sequence, Tuple  # Các type hint

import numpy as np  # Xử lý mảng numpy
import torch  # Framework deep learning PyTorch
import torch.nn.functional as F  # Các hàm trong PyTorch
from scipy import ndimage  # Xử lý ảnh scipy
from scipy.spatial.distance import cdist  # Tính khoảng cách giữa các điểm

try:
    from monai.inferers import sliding_window_inference  # Suy diễn trên các cửa sổ trượt
    from monai.networks.nets import SegResNet  # Mạng SegResNet từ MONAI
except Exception:  # pragma: no cover - optional runtime dependency guard
    sliding_window_inference = None  # Không có MONAI
    SegResNet = None  # Không có MONAI


# Đường dẫn mặc định cho file cấu hình tuning
DEFAULT_TUNING_PATH = Path(
    os.getenv(
        "SEGMENTATION_TUNING_PATH",
        str(Path(__file__).resolve().parents[1] / "checkpoints" / "benchmark" / "segmentation_tuning.json"),
    )
)

# Thư mục source DA-nnUNet trong external (ưu tiên sử dụng)
DEFAULT_DA_NNUNET_SOURCE_DIR = Path(
    os.getenv(
        "DA_NNUNET_SOURCE_DIR",
        str(Path(__file__).resolve().parents[1] / "external" / "DA_nnUNet"),
    )
)

# Tên thư mục con chứa cấu hình DA-nnUNet
_DA_DATASET_DIR = "Dataset142_BraTS2023_MIXED_GRL"
_DA_TRAINER_DIR = "nnUNetTrainerDA_500ep_noDS_4Convs__nnUNetPlans__3d_fullres_bs4"

# Đường dẫn mặc định cho model DA-nnUNet (auto-detect từ checkpoints/ hoặc external/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_checkpoints_da_candidate = (
    _PROJECT_ROOT / "checkpoints" / "da_nnunet" / "DA_nnUNet"
    / _DA_DATASET_DIR / _DA_TRAINER_DIR
)
_external_da_candidate = (
    DEFAULT_DA_NNUNET_SOURCE_DIR / "nnUNet_results"
    / _DA_DATASET_DIR / _DA_TRAINER_DIR
)
DEFAULT_DA_NNUNET_MODEL_DIR = Path(
    os.getenv(
        "DA_NNUNET_MODEL_DIR",
        str(
            _checkpoints_da_candidate if _checkpoints_da_candidate.exists()
            else _external_da_candidate if _external_da_candidate.exists()
            else _checkpoints_da_candidate  # fallback cho thông báo lỗi
        ),
    )
)
# Các phương thức MRI được sử dụng trong BraTS
BRATS_INPUT_MODALITIES = ("t1c", "t1", "t2", "flair")
# Ánh xạ các alias của phương thức MRI
BRATS_MODALITY_ALIASES = {
    "t1c": {"t1c", "t1ce", "t1gd", "contrast", "postcontrast", "post_gad"},  # T1 tương phản
    "t1": {"t1"},  # T1 không tương phản
    "t2": {"t2"},  # T2
    "flair": {"flair", "t2f", "fla", "t2flair"},  # FLAIR
}


def _runtime_mode() -> str:
    mode = str(os.getenv("AI_RUNTIME_MODE", "full")).strip().lower()
    return mode if mode in {"full", "fast"} else "full"


def _select_flip_views(
    flip_views: Sequence[Tuple[int, ...]],
    spatial_shape: Sequence[int],
) -> Sequence[Tuple[int, ...]]:
    if _runtime_mode() == "fast":
        return [(), (2,), (3,)]
    voxel_count = int(np.prod([max(int(dim), 1) for dim in spatial_shape]))
    if voxel_count >= (224 * 224 * 160):
        return [(), (2,), (3,)]
    return flip_views


def clamp(value: float, lower: float, upper: float) -> float:
    """Giới hạn giá trị trong khoảng [lower, upper]"""
    return max(lower, min(upper, value))  # Đảm bảo value nằm trong [lower, upper]


def _load_segmentation_tuning(path: Optional[Path] = None) -> Dict[str, Any]:
    """Tải cấu hình tuning từ file JSON"""
    tuning_path = Path(path or DEFAULT_TUNING_PATH)  # Đường dẫn file
    if not tuning_path.exists():  # Nếu file không tồn tại
        return {}  # Trả về dict rỗng
    try:
        payload = json.loads(tuning_path.read_text(encoding="utf-8"))  # Đọc JSON
        if isinstance(payload, dict):  # Nếu là dict
            payload["_path"] = str(tuning_path)  # Thêm đường dẫn
            return payload  # Trả về
    except Exception:
        return {}  # Trả về rỗng nếu lỗi
    return {}


def _tuning_for_model(model_key: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Lấy cấu hình tuning cho một model cụ thể"""
    tuning = payload or _load_segmentation_tuning()  # Tải cấu hình hoặc dùng payload
    best_configs = tuning.get("best_configs", {}) if isinstance(tuning, dict) else {}  # Lấy best configs
    selected = best_configs.get(model_key, {}) if isinstance(best_configs, dict) else {}  # Lấy config cho model
    if not isinstance(selected, dict):  # Nếu không phải dict
        return {}  # Trả về rỗng
    resolved = dict(selected)  # Copy dict
    if "_path" in tuning:  # Nếu có đường dẫn
        resolved["_path"] = tuning["_path"]  # Thêm đường dẫn
    return resolved  # Trả về


@dataclass
class SegmentationPrediction:
    """Kết quả dự đoán phân đoạn"""
    model_key: str  # Khóa model
    model_name: str  # Tên model
    mask: np.ndarray  # Mặt nạ nhị phân (0: nền, 1: u)
    probability: np.ndarray  # Bản đồ xác suất
    threshold: float  # Ngưỡng quyết định
    quality_score: float  # Điểm chất lượng
    confidence_score: float  # Điểm tin cậy
    inference_mode: str  # Chế độ suy diễn
    input_shape: Tuple[int, ...]  # Kích thước input
    metadata: Dict[str, Any] = field(default_factory=dict)  # Siêu dữ liệu


class BaseSegmentationEngine:
    """Base class cho các engine phân đoạn"""
    model_key = "unknown"  # Khóa model
    model_name = "Unknown model"  # Tên model

    def is_ready(self) -> bool:
        """Kiểm tra engine có sẵn sàng không"""
        return False  # Mặc định không sẵn sàng

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        """Phân đoạn volume MRI"""
        raise NotImplementedError  # Phải override


def _normalize_channel(volume: np.ndarray) -> np.ndarray:
    """Chuẩn hóa channel ảnh (standardization)"""
    image = np.asarray(volume, dtype=np.float32)  # Chuyển thành float32
    finite_mask = np.isfinite(image)  # Tìm các giá trị hữu hạn
    if not finite_mask.any():  # Nếu không có giá trị hữu hạn
        return np.zeros_like(image, dtype=np.float32)  # Trả về zeros

    image = np.where(finite_mask, image, 0.0)  # Thay giá trị không hữu hạn bằng 0
    nonzero = image[np.abs(image) > 1e-6]  # Lấy giá trị khác 0
    sample = nonzero if nonzero.size else image[finite_mask]  # Chọn mẫu
    lower, upper = np.percentile(sample, [0.5, 99.5])  # Tính percentile
    image = np.clip(image, lower, upper)  # Cắt giá trị ngoài khoảng

    focus = image[np.abs(image) > 1e-6]  # Lấy giá trị khác 0
    mean = float(focus.mean()) if focus.size else float(image.mean())  # Tính trung bình
    std = float(focus.std()) if focus.size else float(image.std())  # Tính độ lệch chuẩn
    return ((image - mean) / max(std, 1e-6)).astype(np.float32)  # Chuẩn hóa


def _ensure_channel_first(volume: np.ndarray) -> np.ndarray:
    """Đảm bảo volume có dạng channel-first (C, H, W, D)"""
    array = np.asarray(volume, dtype=np.float32)  # Chuyển thành float32
    if array.ndim == 3:  # Nếu là 3D (H, W, D)
        return array[None, ...]  # Thêm channel: (1, H, W, D)

    if array.ndim != 4:  # Nếu không phải 4D
        raise ValueError(f"Expected a 3D or 4D volume, received shape={array.shape}")  # Ném lỗi

    if array.shape[0] <= 4:  # Nếu chiều đầu ≤ 4 (likely channel-first)
        return array  # Trả về như есть

    if array.shape[-1] <= 4:  # Nếu chiều cuối ≤ 4 (likely channel-last)
        return np.moveaxis(array, -1, 0)  # Chuyển channel về đầu

    return array[:1, ...]  # Lấy channel đầu tiên


def _build_pseudo_brats_channels(volume: np.ndarray) -> np.ndarray:
    """Xây dựng 4 channel giả lập từ 1 channel"""
    base = _normalize_channel(volume)  # Chuẩn hóa
    smooth = ndimage.gaussian_filter(base, sigma=1.0)  # Làm mịn
    detail = base - smooth  # Lấy chi tiết
    t1 = base  # T1 = giá trị gốc
    t1c = np.clip(base * 1.08 + np.maximum(detail, 0.0) * 0.55, -5.0, 5.0)  # T1 với tương phản
    t2 = np.clip(smooth * 0.92 + detail * 0.12, -5.0, 5.0)  # T2
    flair = np.clip(base * 1.18 + np.abs(detail) * 0.33, -5.0, 5.0)  # FLAIR
    return np.stack([t1c, t1, t2, flair], axis=0).astype(np.float32)  # Stack thành 4 channel


def adapt_volume_for_brats(volume: np.ndarray) -> np.ndarray:
    """Điều chỉnh volume để phù hợp với định dạng BraTS"""
    channel_first = _ensure_channel_first(volume)  # Đảm bảo channel-first
    channel_count = channel_first.shape[0]  # Số channel

    if channel_count == 4:  # Nếu đã có 4 channel
        return np.stack([_normalize_channel(channel_first[index]) for index in range(4)], axis=0)  # Chuẩn hóa từng channel

    if channel_count == 1:  # Nếu chỉ 1 channel
        return _build_pseudo_brats_channels(channel_first[0])  # Xây dựng 4 channel giả lập

    if 1 < channel_count < 4:  # Nếu 2-3 channel
        normalized = [_normalize_channel(channel_first[index]) for index in range(channel_count)]  # Chuẩn hóa
        while len(normalized) < 4:  # Cho đến khi có 4 channel
            normalized.append(normalized[-1])  # Sao chép channel cuối cùng
        return np.stack(normalized[:4], axis=0)  # Lấy 4 channel đầu

    return _build_pseudo_brats_channels(channel_first[0])  # Xây dựng từ channel đầu tiên


def _normalize_brats_modality_label(label: str) -> Optional[str]:
    """Chuẩn hóa nhãn phương thức MRI"""
    normalized = "".join(character for character in str(label).strip().lower() if character.isalnum())  # Loại bỏ ký tự đặc biệt
    if not normalized:  # Nếu chuỗi trống
        return None  # Trả về None
    for modality, aliases in BRATS_MODALITY_ALIASES.items():  # Duyệt các ánh xạ
        if normalized in aliases:  # Nếu khớp
            return modality  # Trả về loại phương thức
    return None  # Không khớp


def _prepare_da_nnunet_input(
    volume: np.ndarray,
    series_labels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Chuẩn bị đầu vào cho DA-nnUNet"""
    channel_first = _ensure_channel_first(volume)  # Đảm bảo channel-first
    normalized_channels = [_normalize_channel(channel_first[index]) for index in range(channel_first.shape[0])]  # Chuẩn hóa
    mapped_labels = [_normalize_brats_modality_label(label) for label in (series_labels or ())]  # Ánh xạ nhãn

    if mapped_labels and len(mapped_labels) == len(normalized_channels):  # Nếu có nhãn và số lượng khớp
        by_label: Dict[str, np.ndarray] = {}  # Dict ánh xạ nhãn -> channel
        for label, channel in zip(mapped_labels, normalized_channels):  # Duyệt từng cặp
            if label and label not in by_label:  # Nếu nhãn tồn tại và chưa có
                by_label[label] = channel  # Thêm vào dict
        if all(modality in by_label for modality in BRATS_INPUT_MODALITIES):  # Nếu có đầy đủ các modality
            return (
                np.stack([by_label[modality] for modality in BRATS_INPUT_MODALITIES], axis=0).astype(np.float32),  # Xếp stack theo thứ tự
                tuple(BRATS_INPUT_MODALITIES),  # Trả về nhãn đúng thứ tự
            )

    if channel_first.shape[0] >= 4:  # Nếu có ≥ 4 channel
        return (
            np.stack(normalized_channels[:4], axis=0).astype(np.float32),  # Lấy 4 channel đầu
            tuple(mapped_labels[:4]) if mapped_labels[:4] else tuple(BRATS_INPUT_MODALITIES),  # Trả về nhãn
        )

    raise ValueError("DA-nnUNet requires four aligned MRI channels (T1c, T1, T2, FLAIR)")  # Ném lỗi


def _build_single_channel_for_nnunet(volume: np.ndarray) -> np.ndarray:
    """Xây dựng single channel từ multiple channels cho nnUNet"""
    channel_first = _ensure_channel_first(volume)  # Đảm bảo channel-first
    normalized = [_normalize_channel(channel_first[index]) for index in range(min(channel_first.shape[0], 4))]  # Chuẩn hóa tối đa 4 channel

    if not normalized:  # Nếu không có channel
        return np.zeros(channel_first.shape[1:], dtype=np.float32)  # Trả về zeros

    if len(normalized) == 1:  # Nếu chỉ 1 channel
        return normalized[0].astype(np.float32)  # Trả về channel đó

    if len(normalized) >= 4:  # Nếu ≥ 4 channel
        t1c, t1, t2, flair = normalized[:4]  # Lấy 4 channel
        fused = (0.38 * t1c) + (0.12 * t1) + (0.16 * t2) + (0.34 * flair)  # Hợp nhất với trọng số
        contrast = np.maximum(0.62 * flair + 0.38 * t1c, fused)  # Tương phản
        return _normalize_channel((0.72 * fused) + (0.28 * contrast))  # Chuẩn hóa kết quả

    averaged = np.mean(np.stack(normalized, axis=0), axis=0)  # Trung bình các channel
    support = np.max(np.stack(normalized, axis=0), axis=0)  # Lấy max các channel
    return _normalize_channel((0.68 * averaged) + (0.32 * support))  # Kết hợp


def _foreground_slices_from_raw(
    volume: np.ndarray,
    margin: int = 8,  # Lề bổ sung quanh vùng nền
    min_fraction: float = 0.6,  # Tỉ lệ tối thiểu của vùng nền
) -> Tuple[slice, slice, slice]:
    """Tìm các slice chứa vùng nền (brain) trong volume"""
    channel_first = _ensure_channel_first(volume)  # Đảm bảo channel-first
    foreground = _brain_foreground_mask(channel_first)  # Lấy mask vùng não
    if not np.any(foreground):  # Nếu không có vùng não
        foreground = np.any(np.abs(channel_first) > 1e-6, axis=0)  # Dùng threshold đơn giản

    if not np.any(foreground):  # Nếu vẫn không có
        return tuple(slice(0, int(dim)) for dim in channel_first.shape[1:])  # type: ignore[return-value]  # Trả về full volume

    coords = np.argwhere(foreground)  # Lấy tọa độ foreground
    lower = np.maximum(coords.min(axis=0) - margin, 0)  # Tính cận dưới với lề
    upper = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(channel_first.shape[1:]))  # Tính cận trên

    slices = []
    for axis, (start, stop, full_dim) in enumerate(zip(lower, upper, channel_first.shape[1:])):  # Duyệt từng chiều
        start_i = int(start)  # Cận dưới
        stop_i = int(stop)  # Cận trên
        min_size = max(24, int(full_dim * min_fraction))  # Kích thước tối thiểu
        if (stop_i - start_i) < min_size:  # Nếu kích thước nhỏ hơn tối thiểu
            deficit = min_size - (stop_i - start_i)  # Tính phần cần thêm
            start_i = max(0, start_i - deficit // 2)  # Mở rộng về trước
            stop_i = min(int(full_dim), stop_i + deficit - (deficit // 2))  # Mở rộng về sau
        slices.append(slice(start_i, stop_i))  # Thêm slice

    return slices[0], slices[1], slices[2]  # Trả về 3 slice


def _apply_spatial_crop(volume: np.ndarray, crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    """Cắt volume theo 3 slices"""
    if volume.ndim == 3:  # Nếu 3D
        return volume[crop_slices[0], crop_slices[1], crop_slices[2]]  # Cắt 3D
    if volume.ndim == 4:  # Nếu 4D (channel-first)
        return volume[:, crop_slices[0], crop_slices[1], crop_slices[2]]  # Cắt 4D giữ channel
    raise ValueError(f"Unsupported volume ndim={volume.ndim}")  # Lỗi


def _restore_spatial_crop(array: np.ndarray, full_shape: Sequence[int], crop_slices: Tuple[slice, slice, slice]) -> np.ndarray:
    """Khôi phục volume cắt về kích thước gốc"""
    restored = np.zeros(tuple(int(dim) for dim in full_shape), dtype=array.dtype)  # Tạo zeros với kích thước gốc
    restored[crop_slices[0], crop_slices[1], crop_slices[2]] = array  # Đặt dữ liệu vào vị trí cắt
    return restored  # Trả về


def _pad_spatial_tensor(
    tensor: torch.Tensor,
    multiple: int = 8,  # Pad tới bội số của này
) -> Tuple[torch.Tensor, Tuple[slice, slice, slice], Tuple[int, int, int]]:
    """Padding tensor về kích thước chia hết cho multiple"""
    spatial_shape = [int(dim) for dim in tensor.shape[-3:]]  # Lấy 3 chiều không gian
    pad_widths = []  # Độ rộng padding
    crop_slices = []  # Slice để crop kết quả

    for dim in spatial_shape:  # Duyệt từng chiều
        target = max(multiple, ((dim + multiple - 1) // multiple) * multiple)  # Kích thước target
        total_pad = max(0, target - dim)  # Tổng padding cần
        pad_before = total_pad // 2  # Padding trước
        pad_after = total_pad - pad_before  # Padding sau
        pad_widths.append((pad_before, pad_after))  # Lưu
        crop_slices.append(slice(pad_before, pad_before + dim))  # Slice để lấy phần gốc

    if any(before or after for before, after in pad_widths):  # Nếu cần padding
        tensor = F.pad(
            tensor,
            (  # Thứ tự: D_start, D_end, H_start, H_end, W_start, W_end
                pad_widths[2][0],
                pad_widths[2][1],
                pad_widths[1][0],
                pad_widths[1][1],
                pad_widths[0][0],
                pad_widths[0][1],
            ),
            mode="replicate",  # Dùng replicate padding
        )

    padded_shape = tuple(int(dim) for dim in tensor.shape[-3:])  # Kích thước sau padding
    return tensor, (crop_slices[0], crop_slices[1], crop_slices[2]), padded_shape  # Trả về


def _crop_spatial_tensor(
    tensor: torch.Tensor,
    crop_slices: Tuple[slice, slice, slice],
) -> torch.Tensor:
    """Cắt tensor theo slices"""
    return tensor[..., crop_slices[0], crop_slices[1], crop_slices[2]]  # Cắt 3 chiều cuối cùng


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Lấy thành phần (connected component) lớn nhất từ mask"""
    structure = ndimage.generate_binary_structure(3, 2)  # Cấu trúc kết nối 3D
    labeled, component_count = ndimage.label(mask.astype(bool), structure=structure)  # Label các thành phần
    if component_count <= 1:  # Nếu ≤ 1 thành phần
        return mask.astype(np.uint8), int(component_count), 0  # Trả về nguyên vẹn

    component_sizes = ndimage.sum(mask, labeled, index=np.arange(1, component_count + 1))  # Tính kích thước
    component_sizes = np.asarray(component_sizes, dtype=np.float32)  # Chuyển thành array
    largest = int(component_sizes.argmax()) + 1
    largest_size = float(component_sizes.max()) if component_sizes.size else 0.0
    kept = labeled == largest

    min_kept = max(64.0, largest_size * 0.08)
    for component_index, component_size in enumerate(component_sizes, start=1):
        if component_index == largest:
            continue
        if component_size >= min_kept:
            kept |= labeled == component_index

    removed = int(component_count - len(np.unique(labeled[kept])) + (1 if kept.any() else 0))
    return kept.astype(np.uint8), int(component_count), max(removed, 0)


def _brain_foreground_mask(volume: np.ndarray) -> np.ndarray:
    channel_first = _ensure_channel_first(volume)
    normalized = np.stack(
        [_normalize_channel(channel_first[index]) for index in range(channel_first.shape[0])],
        axis=0,
    ).astype(np.float32)
    energy = np.mean(np.abs(normalized), axis=0)
    energy = ndimage.gaussian_filter(energy, sigma=1.2)

    positive = energy[energy > 1e-6]
    if positive.size == 0:
        return np.any(np.abs(channel_first) > 1e-6, axis=0)

    threshold = float(np.percentile(positive, 30.0) * 0.38)
    threshold = clamp(threshold, 0.025, 0.24)
    mask = energy >= threshold
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    cleaned, _, _ = _largest_component(mask.astype(np.uint8))
    return cleaned.astype(bool)


def _component_support_score(
    component_mask: np.ndarray,
    probability: np.ndarray,
    uncertainty: Optional[np.ndarray],
    reference_image: Optional[np.ndarray],
    support_core: Optional[np.ndarray],
    brain_mask: Optional[np.ndarray],
) -> float:
    component_mask = component_mask.astype(bool)
    size = float(component_mask.sum())
    if size <= 0:
        return 0.0

    mean_probability = float(probability[component_mask].mean())
    peak_probability = float(probability[component_mask].max())
    certainty = 1.0
    if uncertainty is not None:
        certainty = 1.0 - float(np.clip(uncertainty[component_mask].mean() * 6.0, 0.0, 1.0))

    support_overlap = 0.0
    if support_core is not None and np.any(support_core):
        dilated = ndimage.binary_dilation(component_mask, iterations=2)
        support_overlap = float(np.logical_and(dilated, support_core.astype(bool)).sum()) / max(size, 1.0)

    tissue_contrast = 0.0
    if reference_image is not None:
        tissue_contrast = float(np.abs(reference_image[component_mask]).mean())
        tissue_contrast = float(np.clip(tissue_contrast / 3.0, 0.0, 1.0))

    brain_overlap = 1.0
    if brain_mask is not None and np.any(brain_mask):
        brain_overlap = float(np.logical_and(component_mask, brain_mask.astype(bool)).sum()) / max(size, 1.0)

    size_term = float(np.clip(size / 1024.0, 0.0, 1.0))
    score = (
        (mean_probability * 0.36)
        + (peak_probability * 0.16)
        + (certainty * 0.16)
        + (support_overlap * 0.18)
        + (tissue_contrast * 0.08)
        + (size_term * 0.06)
    ) * brain_overlap
    return float(score)


def _retain_salient_components(
    mask: np.ndarray,
    probability: np.ndarray,
    uncertainty: Optional[np.ndarray],
    reference_image: Optional[np.ndarray],
    support_core: Optional[np.ndarray],
    brain_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int]]:
    structure = ndimage.generate_binary_structure(3, 2)
    labeled, component_count = ndimage.label(mask.astype(bool), structure=structure)
    if component_count <= 1:
        return mask.astype(np.uint8), {
            "connected_components": int(component_count),
            "removed_components": 0,
            "retained_components": int(component_count),
        }

    scored_components = []
    for component_index in range(1, component_count + 1):
        component_mask = labeled == component_index
        component_size = int(component_mask.sum())
        if component_size <= 12:
            continue
        score = _component_support_score(
            component_mask=component_mask,
            probability=probability,
            uncertainty=uncertainty,
            reference_image=reference_image,
            support_core=support_core,
            brain_mask=brain_mask,
        )
        peak_probability = float(probability[component_mask].max()) if component_size else 0.0
        scored_components.append((component_index, component_size, score, peak_probability))

    if not scored_components:
        cleaned, component_count, removed = _largest_component(mask.astype(np.uint8))
        return cleaned.astype(np.uint8), {
            "connected_components": int(component_count),
            "removed_components": int(removed),
            "retained_components": max(1, int(component_count - removed)),
        }

    scored_components.sort(key=lambda item: (item[2], item[3], item[1]), reverse=True)
    best_score = max(scored_components[0][2], 1e-6)
    best_peak = max(scored_components[0][3], 1e-6)
    retained = np.zeros_like(mask, dtype=bool)
    kept_count = 0

    for component_index, component_size, score, peak_probability in scored_components:
        if component_size < 24 and peak_probability < 0.60:
            continue
        if kept_count == 0 or score >= (best_score * 0.58) or peak_probability >= (best_peak * 0.86):
            retained |= labeled == component_index
            kept_count += 1

    if not np.any(retained):
        retained |= labeled == scored_components[0][0]
        kept_count = 1

    return retained.astype(np.uint8), {
        "connected_components": int(component_count),
        "removed_components": int(component_count - kept_count),
        "retained_components": int(kept_count),
    }


def _postprocess_mask(
    mask: np.ndarray,
    probability: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    reference_image: Optional[np.ndarray] = None,
    support_core: Optional[np.ndarray] = None,
    brain_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    structure = ndimage.generate_binary_structure(3, 2)
    processed = ndimage.binary_closing(mask.astype(bool), structure=structure, iterations=1)
    processed = ndimage.binary_fill_holes(processed)
    processed = ndimage.binary_opening(processed, structure=structure, iterations=1)
    if probability is not None:
        cleaned, cleanup = _retain_salient_components(
            processed.astype(np.uint8),
            probability=probability.astype(np.float32),
            uncertainty=None if uncertainty is None else uncertainty.astype(np.float32),
            reference_image=None if reference_image is None else reference_image.astype(np.float32),
            support_core=None if support_core is None else support_core.astype(np.uint8),
            brain_mask=None if brain_mask is None else brain_mask.astype(np.uint8),
        )
    else:
        cleaned, component_count, removed = _largest_component(processed.astype(np.uint8))
        cleanup = {
            "connected_components": int(component_count),
            "removed_components": int(removed),
            "retained_components": max(1, int(component_count - removed)),
        }

    if int(cleaned.sum()) == 0 and int(mask.sum()) > 0:
        if probability is not None:
            cleaned, cleanup = _retain_salient_components(
                mask.astype(np.uint8),
                probability=probability.astype(np.float32),
                uncertainty=None if uncertainty is None else uncertainty.astype(np.float32),
                reference_image=None if reference_image is None else reference_image.astype(np.float32),
                support_core=None if support_core is None else support_core.astype(np.uint8),
                brain_mask=None if brain_mask is None else brain_mask.astype(np.uint8),
            )
        else:
            cleaned, component_count, removed = _largest_component(mask.astype(np.uint8))
            cleanup = {
                "connected_components": int(component_count),
                "removed_components": int(removed),
                "retained_components": max(1, int(component_count - removed)),
            }

    return cleaned.astype(np.uint8), cleanup


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    volume = int(a.sum()) + int(b.sum())
    if volume == 0:
        return 1.0
    intersection = int(np.logical_and(a, b).sum())
    return float((2.0 * intersection) / volume)


def _iou_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(a, b).sum())
    return float(intersection / union)


def _sample_points(points: np.ndarray, max_points: int = 2048) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def _surface_hausdorff(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    boundary_a = np.argwhere(ndimage.binary_erosion(mask_a.astype(bool)) ^ mask_a.astype(bool))
    boundary_b = np.argwhere(ndimage.binary_erosion(mask_b.astype(bool)) ^ mask_b.astype(bool))

    if len(boundary_a) == 0 or len(boundary_b) == 0:
        return 0.0

    boundary_a = _sample_points(boundary_a)
    boundary_b = _sample_points(boundary_b)
    distances = cdist(boundary_a.astype(np.float32), boundary_b.astype(np.float32))
    forward = distances.min(axis=1).max()
    backward = distances.min(axis=0).max()
    return float(max(forward, backward))


def _mean_pairwise_dice(masks: Sequence[np.ndarray]) -> float:
    valid = [mask.astype(np.uint8) for mask in masks if mask is not None]
    if len(valid) <= 1:
        return 1.0

    scores = []
    for first in range(len(valid)):
        for second in range(first + 1, len(valid)):
            scores.append(_dice_score(valid[first], valid[second]))
    return float(np.mean(scores)) if scores else 1.0


def _reference_image(volume: np.ndarray) -> np.ndarray:
    channel_first = _ensure_channel_first(volume)
    normalized = [_normalize_channel(channel_first[index]) for index in range(min(channel_first.shape[0], 4))]
    if not normalized:
        return np.zeros(channel_first.shape[1:], dtype=np.float32)
    if len(normalized) == 1:
        return normalized[0].astype(np.float32)
    return np.mean(np.stack(normalized, axis=0), axis=0).astype(np.float32)


def _edge_aware_refine_mask(
    probability: np.ndarray,
    threshold: float,
    reference_image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    base_mask = (probability >= threshold).astype(np.uint8)
    if int(base_mask.sum()) == 0:
        return base_mask, {"edge_alignment_score": 0.0, "edge_refine_gain": 0.0}

    smoothed = ndimage.gaussian_filter(reference_image.astype(np.float32), sigma=1.0)
    gradient = ndimage.gaussian_gradient_magnitude(smoothed, sigma=1.0)
    scale = float(np.percentile(gradient, 99.0)) if np.any(gradient > 0) else 1.0
    gradient = np.clip(gradient / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)

    boundary_band = ndimage.binary_dilation(base_mask.astype(bool), iterations=2) & ~ndimage.binary_erosion(base_mask.astype(bool), iterations=1)
    candidate_region = boundary_band & (probability >= max(threshold - 0.08, 0.20))
    core_region = probability >= min(threshold + 0.10, 0.82)
    snapped_probability = np.clip((probability * 0.87) + (gradient * 0.13), 0.0, 1.0)

    refined_mask = base_mask.astype(bool)
    refined_mask[candidate_region] = snapped_probability[candidate_region] >= max(threshold - 0.03, 0.18)
    refined_mask |= core_region
    refined_mask = ndimage.binary_closing(refined_mask, iterations=1)
    refined_mask = ndimage.binary_fill_holes(refined_mask)

    refined = refined_mask.astype(np.uint8)
    refined_sum = int(refined.sum())
    base_sum = int(base_mask.sum())
    gain = float(refined_sum - base_sum) / max(float(base_sum), 1.0)
    boundary = ndimage.morphological_gradient(refined.astype(np.uint8), size=(3, 3, 3)) > 0
    edge_alignment = float(gradient[boundary].mean()) if np.any(boundary) else 0.0
    return refined, {
        "edge_alignment_score": edge_alignment,
        "edge_refine_gain": gain,
    }


class MonaiBraTSSegmentationEngine(BaseSegmentationEngine):
    model_key = "monai_brats_mri_segmentation"
    model_name = "MONAI BraTS SegResNet Bundle"

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.flip_views = [(), (2,), (3,), (4,), (2, 3)]
        self.tuning = _tuning_for_model(self.model_key)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if SegResNet is None or sliding_window_inference is None:
            raise RuntimeError("MONAI is required to run the BraTS segmentation engine")

        network = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=4,
            out_channels=3,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
        state_dict = torch.load(self.model_path, map_location="cpu")
        network.load_state_dict(state_dict)
        network.to(self.device)
        network.eval()
        self.model = network

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def is_ready(self) -> bool:
        return self.model is not None

    def _roi_size_for(self, spatial_shape: Sequence[int]) -> Tuple[int, int, int]:
        target = (160, 160, 128)
        roi = []
        for dim, limit in zip(spatial_shape, target):
            raw = min(int(dim), limit)
            aligned = max(8, (raw // 8) * 8)
            if aligned == 0:
                aligned = 8
            roi.append(min(int(dim), aligned if aligned <= int(dim) else int(dim)))
        return tuple(roi)

    def _resize_map(self, tensor: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
        if tuple(int(dim) for dim in tensor.shape[-3:]) == tuple(int(dim) for dim in spatial_shape):
            return tensor
        resized = F.interpolate(
            tensor[None],
            size=tuple(int(dim) for dim in spatial_shape),
            mode="trilinear",
            align_corners=False,
        )
        return resized[0]

    def _predict_probabilities(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Segmentation engine is not initialized")
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self._roi_size_for(input_tensor.shape[-3:]),
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.5,
        )
        probabilities = torch.sigmoid(logits)[0]
        return probabilities

    def _tta_probabilities(
        self,
        input_tensor: torch.Tensor,
        crop_slices: Tuple[slice, slice, slice],
        flip_views: Sequence[Tuple[int, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = []
        with torch.no_grad():
            for axes in flip_views:
                augmented = torch.flip(input_tensor, dims=list(axes)) if axes else input_tensor
                probabilities = self._predict_probabilities(augmented)
                if axes:
                    probabilities = torch.flip(probabilities, dims=[axis - 1 for axis in axes])
                probabilities = _crop_spatial_tensor(probabilities, crop_slices)
                predictions.append(probabilities)

        stacked = torch.stack(predictions, dim=0)
        return stacked, stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)

    def _compose_probability(self, probability_map: np.ndarray) -> np.ndarray:
        tumor_core = probability_map[0]
        whole_tumor = probability_map[1]
        enhancing = probability_map[2]
        support = np.maximum(tumor_core, enhancing)
        composite = np.maximum(whole_tumor, support * 0.78)
        return np.clip(composite, 0.0, 1.0).astype(np.float32)

    def _dynamic_threshold(self, probability: np.ndarray, uncertainty: np.ndarray) -> float:
        candidate_region = probability[probability >= 0.35]
        support = float(candidate_region.mean()) if candidate_region.size else float(probability.max())
        uncertainty_level = float(np.mean(uncertainty))
        threshold = 0.50 + uncertainty_level * 0.10 - max(0.0, support - 0.55) * 0.12
        threshold += float(self.tuning.get("threshold_bias", 0.0) or 0.0)
        if self.tuning.get("threshold") is not None:
            threshold = (threshold * 0.45) + (float(self.tuning["threshold"]) * 0.55)
        return clamp(threshold, 0.42, 0.58)

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        if self.model is None:
            raise RuntimeError("Segmentation engine is not initialized")

        foreground_slices = _foreground_slices_from_raw(volume)
        cropped_volume = _apply_spatial_crop(np.asarray(volume, dtype=np.float32), foreground_slices)
        channels = adapt_volume_for_brats(cropped_volume)
        input_tensor = torch.from_numpy(channels[None]).to(self.device)
        original_shape = tuple(int(dim) for dim in channels.shape[1:])
        padded_tensor, pad_crop_slices, padded_shape = _pad_spatial_tensor(input_tensor, multiple=8)
        flip_views = _select_flip_views(self.flip_views, original_shape)
        stacked_probability_tensor, mean_probability_tensor, variance_tensor = self._tta_probabilities(
            padded_tensor,
            pad_crop_slices,
            flip_views,
        )
        mean_probability_tensor = self._resize_map(mean_probability_tensor, original_shape)
        variance_tensor = self._resize_map(variance_tensor, original_shape)

        mean_probability = mean_probability_tensor.cpu().numpy().astype(np.float32)
        variance_map = variance_tensor.cpu().numpy().astype(np.float32)
        composite_probability = self._compose_probability(mean_probability)
        uncertainty_map = np.clip(variance_map.mean(axis=0), 0.0, 1.0).astype(np.float32)
        reference_image = _reference_image(cropped_volume)
        brain_mask = _brain_foreground_mask(cropped_volume)

        threshold = self._dynamic_threshold(composite_probability, uncertainty_map)
        raw_mask = (composite_probability >= threshold).astype(np.uint8)
        if int(raw_mask.sum()) < 64:
            threshold = 0.42
            raw_mask = (composite_probability >= threshold).astype(np.uint8)

        refined_mask, edge_meta = _edge_aware_refine_mask(composite_probability, threshold, reference_image)
        if int(refined_mask.sum()) > 0:
            raw_mask = refined_mask

        support_core = composite_probability >= min(threshold + 0.12, 0.90)
        mask, cleanup = _postprocess_mask(
            raw_mask,
            probability=composite_probability,
            uncertainty=uncertainty_map,
            reference_image=reference_image,
            support_core=support_core,
            brain_mask=brain_mask,
        )
        if int(mask.sum()) == 0 and int(raw_mask.sum()) > 0:
            mask = raw_mask

        vote_probabilities = stacked_probability_tensor.cpu().numpy().astype(np.float32)
        vote_masks = []
        for vote_probability in vote_probabilities:
            vote_composite = self._compose_probability(vote_probability)
            vote_masks.append((vote_composite >= threshold).astype(np.uint8))

        internal_dice = float(np.mean([_dice_score(vote_mask, mask) for vote_mask in vote_masks])) if vote_masks else 0.0
        internal_iou = float(np.mean([_iou_score(vote_mask, mask) for vote_mask in vote_masks])) if vote_masks else 0.0
        cleanup_hausdorff = _surface_hausdorff(raw_mask, mask) if _runtime_mode() == "full" else 0.0

        positive_region = composite_probability[mask > 0]
        region_mean = float(positive_region.mean()) if positive_region.size else float(composite_probability.max())
        entropy = -(
            composite_probability * np.log(np.clip(composite_probability, 1e-6, 1.0))
            + (1.0 - composite_probability) * np.log(np.clip(1.0 - composite_probability, 1e-6, 1.0))
        )
        normalized_entropy = float(entropy.mean() / np.log(2.0))
        stability = 1.0 - float(np.clip(uncertainty_map.mean() * 6.0, 0.0, 1.0))
        confidence = clamp((region_mean * 76.0) + (stability * 18.0) - (normalized_entropy * 10.0), 62.0, 99.0)
        quality = clamp((confidence / 100.0) * 0.70 + stability * 0.18 + (1.0 - normalized_entropy) * 0.08, 0.60, 0.98)

        full_shape = tuple(int(dim) for dim in _ensure_channel_first(volume).shape[1:])
        if original_shape != full_shape:
            mask = _restore_spatial_crop(mask.astype(np.uint8), full_shape, foreground_slices)
            composite_probability = _restore_spatial_crop(composite_probability.astype(np.float32), full_shape, foreground_slices)

        return SegmentationPrediction(
            model_key=self.model_key,
            model_name=self.model_name,
            mask=mask.astype(np.uint8),
            probability=composite_probability.astype(np.float32),
            threshold=float(threshold),
            quality_score=float(quality),
            confidence_score=float(confidence),
            inference_mode="MONAI BraTS SegResNet + hop nhat TTA",
            input_shape=tuple(int(dim) for dim in channels.shape[1:]),
            metadata={
                "device": str(self.device),
                "input_channels": int(channels.shape[0]),
                "original_shape": list(original_shape),
                "padded_shape": list(padded_shape),
                "foreground_crop_bbox": [
                    [int(foreground_slices[0].start), int(foreground_slices[0].stop)],
                    [int(foreground_slices[1].start), int(foreground_slices[1].stop)],
                    [int(foreground_slices[2].start), int(foreground_slices[2].stop)],
                ],
                "whole_tumor_channel": 1,
                "tta_views": int(len(flip_views)),
                "runtime_mode": _runtime_mode(),
                "tuning_applied": bool(self.tuning),
                "tuning_source": self.tuning.get("_path"),
                "mean_uncertainty": float(np.mean(uncertainty_map)),
                "stability_score": float(stability),
                "internal_dice_score": internal_dice,
                "internal_iou_score": internal_iou,
                "cleanup_hausdorff_voxels": cleanup_hausdorff,
                **edge_meta,
                **cleanup,
            },
        )


class _ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.01),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _EncoderBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = _ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _DecoderBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, up_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = torch.nn.ConvTranspose3d(in_channels, up_channels, kernel_size=2, stride=2)
        self.conv = _ConvBlock3D(up_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _NnUnet3DStrongNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = _ConvBlock3D(1, 24)
        self.encoder2 = _EncoderBlock3D(24, 48)
        self.encoder3 = _EncoderBlock3D(48, 96)
        self.encoder4 = _EncoderBlock3D(96, 192)
        self.bottleneck = _EncoderBlock3D(192, 240)
        self.decoder1 = _DecoderBlock3D(240, 120, 192, 192)
        self.decoder2 = _DecoderBlock3D(192, 96, 96, 96)
        self.decoder3 = _DecoderBlock3D(96, 48, 48, 48)
        self.decoder4 = _DecoderBlock3D(48, 24, 24, 24)
        self.head = torch.nn.Conv3d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)
        dec1 = self.decoder1(bottleneck, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)
        return self.head(dec4)


class NnUnet3DStrongEngine(BaseSegmentationEngine):
    model_key = "nnunet3d_strong"
    model_name = "nnU-Net 3D Strong"

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = MonaiBraTSSegmentationEngine._resolve_device(device)
        self.model: Optional[torch.nn.Module] = None
        self.flip_views = [(), (2,), (3,), (4,), (2, 3)]
        self.patch_size = (64, 128, 128)
        self.tuning = _tuning_for_model(self.model_key)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if sliding_window_inference is None:
            raise RuntimeError("MONAI sliding_window_inference is required to run nnU-Net inference")

        payload = torch.load(self.model_path, map_location="cpu")
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else payload
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        patch_size = config.get("patch_size")
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 3:
            self.patch_size = tuple(int(item) for item in patch_size)

        network = _NnUnet3DStrongNet()
        network.load_state_dict(state_dict, strict=True)
        network.to(self.device)
        network.eval()
        self.model = network

    def is_ready(self) -> bool:
        return self.model is not None

    def _roi_size_for(self, spatial_shape: Sequence[int]) -> Tuple[int, int, int]:
        roi = []
        for dim, limit in zip(spatial_shape, self.patch_size):
            raw = min(int(dim), int(limit))
            aligned = max(16, (raw // 16) * 16)
            roi.append(min(int(dim), aligned))
        return tuple(roi)

    def _predict_probability(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Segmentation engine is not initialized")
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self._roi_size_for(input_tensor.shape[-3:]),
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.5,
        )
        return torch.sigmoid(logits)[0, 0]

    def _tta_probabilities(
        self,
        input_tensor: torch.Tensor,
        crop_slices: Tuple[slice, slice, slice],
        flip_views: Sequence[Tuple[int, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = []
        with torch.no_grad():
            for axes in flip_views:
                augmented = torch.flip(input_tensor, dims=list(axes)) if axes else input_tensor
                probability = self._predict_probability(augmented)
                if axes:
                    probability = torch.flip(probability, dims=[axis - 2 for axis in axes])
                probability = _crop_spatial_tensor(probability, crop_slices)
                predictions.append(probability)

        stacked = torch.stack(predictions, dim=0)
        return stacked, stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)

    def _dynamic_threshold(self, probability: np.ndarray, uncertainty: np.ndarray) -> float:
        candidate_region = probability[probability >= 0.30]
        support = float(candidate_region.mean()) if candidate_region.size else float(probability.max())
        uncertainty_level = float(np.mean(uncertainty))
        threshold = 0.46 + uncertainty_level * 0.12 - max(0.0, support - 0.52) * 0.10
        threshold += float(self.tuning.get("threshold_bias", 0.0) or 0.0)
        if self.tuning.get("threshold") is not None:
            threshold = (threshold * 0.45) + (float(self.tuning["threshold"]) * 0.55)
        return clamp(threshold, 0.36, 0.58)

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        if self.model is None:
            raise RuntimeError("Segmentation engine is not initialized")

        crop_bbox = _foreground_slices_from_raw(volume)
        cropped_volume = _apply_spatial_crop(np.asarray(volume, dtype=np.float32), crop_bbox)
        single_channel = _build_single_channel_for_nnunet(cropped_volume)
        input_tensor = torch.from_numpy(single_channel[None, None]).to(self.device)
        cropped_shape = tuple(int(dim) for dim in single_channel.shape)
        padded_tensor, crop_slices, padded_shape = _pad_spatial_tensor(input_tensor, multiple=16)
        flip_views = _select_flip_views(self.flip_views, cropped_shape)
        stacked_probability_tensor, mean_probability_tensor, variance_tensor = self._tta_probabilities(
            padded_tensor,
            crop_slices,
            flip_views,
        )

        mean_probability = mean_probability_tensor.cpu().numpy().astype(np.float32)
        variance_map = variance_tensor.cpu().numpy().astype(np.float32)
        uncertainty_map = np.clip(variance_map, 0.0, 1.0).astype(np.float32)
        reference_image = _reference_image(cropped_volume)
        brain_mask = _brain_foreground_mask(cropped_volume)
        threshold = self._dynamic_threshold(mean_probability, uncertainty_map)
        raw_mask = (mean_probability >= threshold).astype(np.uint8)
        if int(raw_mask.sum()) < 64:
            threshold = 0.34
            raw_mask = (mean_probability >= threshold).astype(np.uint8)

        refined_mask, edge_meta = _edge_aware_refine_mask(mean_probability, threshold, reference_image)
        if int(refined_mask.sum()) > 0:
            raw_mask = refined_mask

        support_core = mean_probability >= min(threshold + 0.12, 0.88)
        mask, cleanup = _postprocess_mask(
            raw_mask,
            probability=mean_probability,
            uncertainty=uncertainty_map,
            reference_image=reference_image,
            support_core=support_core,
            brain_mask=brain_mask,
        )
        if int(mask.sum()) == 0 and int(raw_mask.sum()) > 0:
            mask = raw_mask

        vote_probabilities = stacked_probability_tensor.cpu().numpy().astype(np.float32)
        vote_masks = [(probability >= threshold).astype(np.uint8) for probability in vote_probabilities]
        internal_dice = float(np.mean([_dice_score(vote_mask, mask) for vote_mask in vote_masks])) if vote_masks else 0.0
        internal_iou = float(np.mean([_iou_score(vote_mask, mask) for vote_mask in vote_masks])) if vote_masks else 0.0
        cleanup_hausdorff = _surface_hausdorff(raw_mask, mask) if _runtime_mode() == "full" else 0.0
        stability = 1.0 - float(np.clip(uncertainty_map.mean() * 7.0, 0.0, 1.0))
        confidence = clamp((float(mean_probability[mask > 0].mean()) if np.any(mask > 0) else float(mean_probability.max())) * 74.0 + stability * 19.0, 58.0, 97.0)
        quality = clamp((confidence / 100.0) * 0.66 + stability * 0.24 + internal_dice * 0.08, 0.56, 0.95)

        full_shape = tuple(int(dim) for dim in _ensure_channel_first(volume).shape[1:])
        if cropped_shape != full_shape:
            mask = _restore_spatial_crop(mask.astype(np.uint8), full_shape, crop_bbox)
            mean_probability = _restore_spatial_crop(mean_probability.astype(np.float32), full_shape, crop_bbox)

        return SegmentationPrediction(
            model_key=self.model_key,
            model_name=self.model_name,
            mask=mask.astype(np.uint8),
            probability=mean_probability.astype(np.float32),
            threshold=float(threshold),
            quality_score=float(quality),
            confidence_score=float(confidence),
            inference_mode="nnU-Net 3D Strong + foreground crop + TTA",
            input_shape=full_shape,
            metadata={
                "device": str(self.device),
                "input_channels": 1,
                "original_shape": list(cropped_shape),
                "padded_shape": list(padded_shape),
                "foreground_crop_bbox": [
                    [int(crop_bbox[0].start), int(crop_bbox[0].stop)],
                    [int(crop_bbox[1].start), int(crop_bbox[1].stop)],
                    [int(crop_bbox[2].start), int(crop_bbox[2].stop)],
                ],
                "tta_views": int(len(flip_views)),
                "runtime_mode": _runtime_mode(),
                "tuning_applied": bool(self.tuning),
                "tuning_source": self.tuning.get("_path"),
                "mean_uncertainty": float(np.mean(uncertainty_map)),
                "stability_score": float(stability),
                "internal_dice_score": internal_dice,
                "internal_iou_score": internal_iou,
                "cleanup_hausdorff_voxels": cleanup_hausdorff,
                **edge_meta,
                **cleanup,
            },
        )


class WeightedEnsembleSegmentationEngine(BaseSegmentationEngine):
    model_key = "hybrid_segmentation_ensemble"
    model_name = "MONAI + nnU-Net weighted ensemble"

    def __init__(self, engines: Sequence[BaseSegmentationEngine]):
        self.engines = [engine for engine in engines if engine is not None and engine.is_ready()]
        if not self.engines:
            raise RuntimeError("No ready segmentation engines for ensemble")
        self.tuning = _tuning_for_model(self.model_key)

    def is_ready(self) -> bool:
        return bool(self.engines)

    def _base_weight(self, prediction: SegmentationPrediction, channel_count: int) -> float:
        if prediction.model_key == MonaiBraTSSegmentationEngine.model_key and self.tuning.get("monai_weight") is not None:
            return float(self.tuning["monai_weight"])
        if prediction.model_key == NnUnet3DStrongEngine.model_key and self.tuning.get("nnunet_weight") is not None:
            return float(self.tuning["nnunet_weight"])
        if prediction.model_key == MonaiBraTSSegmentationEngine.model_key:
            if channel_count >= 4:
                return 0.76
            if channel_count >= 2:
                return 0.66
            return 0.56
        if prediction.model_key == NnUnet3DStrongEngine.model_key:
            if channel_count >= 4:
                return 0.24
            if channel_count >= 2:
                return 0.34
            return 0.44
        return 0.18

    def _dynamic_threshold(self, probability: np.ndarray, model_uncertainty: np.ndarray, agreement: float) -> float:
        candidate_region = probability[probability >= 0.34]
        support = float(candidate_region.mean()) if candidate_region.size else float(probability.max())
        uncertainty_level = float(np.mean(model_uncertainty))
        threshold = 0.47 + uncertainty_level * 0.12 - max(0.0, support - 0.56) * 0.10 - max(0.0, agreement - 0.75) * 0.04
        threshold += float(self.tuning.get("threshold_bias", 0.0) or 0.0)
        if self.tuning.get("threshold") is not None:
            threshold = (threshold * 0.42) + (float(self.tuning["threshold"]) * 0.58)
        return clamp(threshold, 0.38, 0.58)

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        predictions = [
            engine.segment(volume, voxel_spacing=voxel_spacing, series_labels=series_labels)
            for engine in self.engines
        ]
        channel_count = int(_ensure_channel_first(volume).shape[0])
        probabilities = np.stack([prediction.probability.astype(np.float32) for prediction in predictions], axis=0)
        model_masks = [(prediction.probability >= prediction.threshold).astype(np.uint8) for prediction in predictions]
        agreement = _mean_pairwise_dice(model_masks)
        base_weights = np.asarray([self._base_weight(prediction, channel_count) for prediction in predictions], dtype=np.float32)
        reference_index = int(base_weights.argmax())
        reference_mask = model_masks[reference_index]
        consistencies = np.asarray(
            [_dice_score(model_mask, reference_mask) if index != reference_index else 1.0 for index, model_mask in enumerate(model_masks)],
            dtype=np.float32,
        )

        raw_weights = np.asarray(
            [
                base_weights[index]
                * (0.18 + (consistencies[index] * 0.82))
                * (0.72 + (prediction.quality_score * 0.18) + ((prediction.confidence_score / 100.0) * 0.10))
                for index, prediction in enumerate(predictions)
            ],
            dtype=np.float32,
        )
        weights = raw_weights / np.clip(raw_weights.sum(), 1e-6, None)
        weighted_probability = np.tensordot(weights, probabilities, axes=(0, 0)).astype(np.float32)
        consensus_map = np.mean(np.stack(model_masks, axis=0), axis=0).astype(np.float32)
        consensus_weight = float(self.tuning.get("consensus_weight", 0.18 if agreement >= 0.35 else 0.08))
        consensus_weight = clamp(consensus_weight, 0.0, 0.22)
        blended_probability = np.clip((weighted_probability * (1.0 - consensus_weight)) + (consensus_map * consensus_weight), 0.0, 1.0).astype(np.float32)
        model_uncertainty = probabilities.var(axis=0).astype(np.float32)
        reference_image = _reference_image(volume)
        brain_mask = _brain_foreground_mask(volume)

        threshold = self._dynamic_threshold(blended_probability, model_uncertainty, agreement)
        raw_mask = (blended_probability >= threshold).astype(np.uint8)
        if int(raw_mask.sum()) < 64:
            threshold = 0.36
            raw_mask = (blended_probability >= threshold).astype(np.uint8)

        refined_mask, edge_meta = _edge_aware_refine_mask(blended_probability, threshold, reference_image)
        if int(refined_mask.sum()) > 0:
            raw_mask = refined_mask

        support_core = blended_probability >= min(threshold + 0.12, 0.88)
        mask, cleanup = _postprocess_mask(
            raw_mask,
            probability=blended_probability,
            uncertainty=model_uncertainty,
            reference_image=reference_image,
            support_core=support_core,
            brain_mask=brain_mask,
        )
        if int(mask.sum()) == 0 and int(raw_mask.sum()) > 0:
            mask = raw_mask

        internal_dice = float(np.mean([_dice_score(model_mask, mask) for model_mask in model_masks])) if model_masks else 0.0
        internal_iou = float(np.mean([_iou_score(model_mask, mask) for model_mask in model_masks])) if model_masks else 0.0
        cleanup_hausdorff = _surface_hausdorff(raw_mask, mask)
        stability = 1.0 - float(np.clip(model_uncertainty.mean() * 8.0, 0.0, 1.0))
        weighted_confidence = float(np.dot(weights, np.asarray([prediction.confidence_score for prediction in predictions], dtype=np.float32)))
        weighted_quality = float(np.dot(weights, np.asarray([prediction.quality_score for prediction in predictions], dtype=np.float32)))
        confidence = clamp(weighted_confidence * 0.78 + agreement * 18.0 + stability * 6.0, 66.0, 99.0)
        quality = clamp(weighted_quality * 0.72 + agreement * 0.18 + stability * 0.08, 0.64, 0.99)

        return SegmentationPrediction(
            model_key=self.model_key,
            model_name=self.model_name,
            mask=mask.astype(np.uint8),
            probability=blended_probability.astype(np.float32),
            threshold=float(threshold),
            quality_score=float(quality),
            confidence_score=float(confidence),
            inference_mode="Weighted ensemble: MONAI SegResNet + nnU-Net 3D Strong",
            input_shape=tuple(int(dim) for dim in mask.shape),
            metadata={
                "component_models": [prediction.model_name for prediction in predictions],
                "component_keys": [prediction.model_key for prediction in predictions],
                "component_thresholds": [float(prediction.threshold) for prediction in predictions],
                "component_weights": [float(weight) for weight in weights],
                "component_consistency": [float(value) for value in consistencies],
                "input_channel_count": int(channel_count),
                "tuning_applied": bool(self.tuning),
                "tuning_source": self.tuning.get("_path"),
                "consensus_weight": float(consensus_weight),
                "model_agreement_score": float(agreement),
                "mean_uncertainty": float(model_uncertainty.mean()),
                "stability_score": float(stability),
                "internal_dice_score": float(internal_dice),
                "internal_iou_score": float(internal_iou),
                "cleanup_hausdorff_voxels": float(cleanup_hausdorff),
                **edge_meta,
                **cleanup,
            },
        )


class DaNnUnetSegmentationEngine(BaseSegmentationEngine):
    model_key = "da_nnunet_pediatric"
    model_name = "DA-nnUNet Pediatric Domain-Adapted"

    def __init__(self, model_dir: Path, device: str = "auto", folds: Optional[Sequence[int]] = None):
        self.model_dir = Path(model_dir)
        self.device = MonaiBraTSSegmentationEngine._resolve_device(device)
        self.predictor = None
        self._predictor_lock = threading.Lock()
        self.summary: Dict[str, Any] = {}
        self.postprocess_fns = []
        self.postprocess_kwargs = []
        self.tuning = _tuning_for_model(self.model_key)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"DA-nnUNet model folder not found: {self.model_dir}")

        fold_override = os.getenv("DA_NNUNET_FOLDS", "").strip()
        if folds is not None:
            self.folds = tuple(int(item) for item in folds)
        elif fold_override:
            self.folds = tuple(int(item.strip()) for item in fold_override.split(",") if item.strip())
        elif self.device.type == "cuda":
            self.folds = (0, 1, 2, 3, 4)
        else:
            self.folds = (0,)

        postprocess_dir = self.model_dir / "crossval_results_folds_0_1_2_3_4"
        summary_path = postprocess_dir / "summary.json"
        if summary_path.exists():
            try:
                self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                self.summary = {}

        postprocess_path = postprocess_dir / "postprocessing.pkl"
        if postprocess_path.exists():
            try:
                with postprocess_path.open("rb") as handle:
                    self.postprocess_fns, self.postprocess_kwargs = pickle.load(handle)
            except Exception:
                self.postprocess_fns, self.postprocess_kwargs = [], []

    def is_ready(self) -> bool:
        return self.model_dir.exists()

    def can_handle(self, volume: np.ndarray, series_labels: Optional[Sequence[str]] = None) -> bool:
        channel_first = _ensure_channel_first(volume)
        if channel_first.shape[0] < 4:
            return False
        if not series_labels:
            return channel_first.shape[0] >= 4
        mapped = [_normalize_brats_modality_label(label) for label in series_labels]
        return all(modality in mapped for modality in BRATS_INPUT_MODALITIES)

    def _ensure_predictor(self) -> None:
        if self.predictor is not None:
            return

        with self._predictor_lock:
            if self.predictor is not None:
                return

            os.environ.setdefault("nnUNet_raw", str(self.model_dir.parent / "_nnunet_raw"))
            os.environ.setdefault("nnUNet_preprocessed", str(self.model_dir.parent / "_nnunet_preprocessed"))
            os.environ.setdefault("nnUNet_results", str(self.model_dir.parent))

            # Ưu tiên import nnUNet từ external/DA_nnUNet để dùng đúng bản domain-adapted.
            external_source = Path(os.getenv("DA_NNUNET_SOURCE_DIR", str(DEFAULT_DA_NNUNET_SOURCE_DIR)))
            if external_source.exists():
                source_str = str(external_source.resolve())
                if source_str not in sys.path:
                    sys.path.insert(0, source_str)

            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=self.device.type == "cuda",
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                str(self.model_dir),
                use_folds=self.folds,
                checkpoint_name="checkpoint_final.pth",
            )
            self.predictor = predictor

    def _dynamic_threshold(self, probability: np.ndarray) -> float:
        candidate_region = probability[probability >= 0.30]
        support = float(candidate_region.mean()) if candidate_region.size else float(probability.max())
        threshold = 0.48 - max(0.0, support - 0.58) * 0.10
        threshold += float(self.tuning.get("threshold_bias", 0.0) or 0.0)
        if self.tuning.get("threshold") is not None:
            threshold = (threshold * 0.45) + (float(self.tuning["threshold"]) * 0.55)
        return clamp(threshold, 0.36, 0.56)

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        if not self.can_handle(volume, series_labels):
            raise ValueError("DA-nnUNet requires four aligned modalities for reliable inference")

        self._ensure_predictor()
        if self.predictor is None:
            raise RuntimeError("DA-nnUNet predictor is not initialized")

        ordered_volume, ordered_labels = _prepare_da_nnunet_input(volume, series_labels)
        spacing = tuple(float(item) for item in (voxel_spacing or (1.0, 1.0, 1.0)))
        properties = {"spacing": list(spacing)}
        segmentation, probabilities = self.predictor.predict_single_npy_array(
            ordered_volume.astype(np.float32),
            properties,
            None,
            None,
            True,
        )

        label_map = np.asarray(segmentation, dtype=np.uint8)
        if self.postprocess_fns:
            try:
                from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing

                label_map = apply_postprocessing(label_map, self.postprocess_fns, self.postprocess_kwargs).astype(np.uint8)
            except Exception:
                label_map = label_map.astype(np.uint8)

        probabilities_np = np.asarray(probabilities, dtype=np.float32)
        if probabilities_np.ndim == 4:
            if probabilities_np.shape[0] == 1:
                whole_tumor_probability = probabilities_np[0]
            else:
                background_probability = probabilities_np[0]
                tumor_channels = probabilities_np[1:]
                tumor_probability = np.max(tumor_channels, axis=0) if tumor_channels.size else (1.0 - background_probability)
                whole_tumor_probability = np.maximum(tumor_probability, 1.0 - background_probability)
        elif probabilities_np.ndim == 3:
            whole_tumor_probability = probabilities_np
        else:
            whole_tumor_probability = (label_map > 0).astype(np.float32)
        whole_tumor_probability = np.clip(whole_tumor_probability, 0.0, 1.0).astype(np.float32)

        raw_mask = (label_map > 0).astype(np.uint8)
        threshold = self._dynamic_threshold(whole_tumor_probability)
        threshold_mask = (whole_tumor_probability >= threshold).astype(np.uint8)
        if int(threshold_mask.sum()) > 0:
            raw_mask = np.logical_or(raw_mask.astype(bool), threshold_mask.astype(bool)).astype(np.uint8)

        reference_image = _reference_image(ordered_volume)
        brain_mask = _brain_foreground_mask(ordered_volume)
        support_core = whole_tumor_probability >= min(threshold + 0.10, 0.88)
        mask, cleanup = _postprocess_mask(
            raw_mask,
            probability=whole_tumor_probability,
            uncertainty=None,
            reference_image=reference_image,
            support_core=support_core.astype(np.uint8),
            brain_mask=brain_mask.astype(np.uint8),
        )
        if int(mask.sum()) == 0 and int(raw_mask.sum()) > 0:
            mask = raw_mask

        positive_region = whole_tumor_probability[mask > 0]
        region_mean = float(positive_region.mean()) if positive_region.size else float(whole_tumor_probability.max())
        entropy = -(
            whole_tumor_probability * np.log(np.clip(whole_tumor_probability, 1e-6, 1.0))
            + (1.0 - whole_tumor_probability) * np.log(np.clip(1.0 - whole_tumor_probability, 1e-6, 1.0))
        )
        normalized_entropy = float(entropy.mean() / np.log(2.0))
        stability = clamp(1.0 - normalized_entropy, 0.0, 1.0)
        confidence = clamp((region_mean * 82.0) + (stability * 12.0), 64.0, 99.0)
        quality = clamp((confidence / 100.0) * 0.76 + stability * 0.18, 0.62, 0.99)

        summary_metrics = self.summary.get("foreground_mean", {}) if isinstance(self.summary, dict) else {}
        return SegmentationPrediction(
            model_key=self.model_key,
            model_name=self.model_name,
            mask=mask.astype(np.uint8),
            probability=whole_tumor_probability.astype(np.float32),
            threshold=float(threshold),
            quality_score=float(quality),
            confidence_score=float(confidence),
            inference_mode="DA-nnUNet pediatric 3D fullres",
            input_shape=tuple(int(dim) for dim in ordered_volume.shape[1:]),
            metadata={
                "device": str(self.device),
                "input_channels": int(ordered_volume.shape[0]),
                "ordered_modalities": list(ordered_labels),
                "folds": [int(item) for item in self.folds],
                "postprocessing_applied": bool(self.postprocess_fns),
                "runtime_mode": _runtime_mode(),
                "tuning_applied": bool(self.tuning),
                "tuning_source": self.tuning.get("_path"),
                "stability_score": float(stability),
                "mean_uncertainty": float(1.0 - stability),
                "summary_mean_dice": summary_metrics.get("Dice"),
                "summary_mean_iou": summary_metrics.get("IoU"),
                **cleanup,
            },
        )


class ClinicalRoutingSegmentationEngine(BaseSegmentationEngine):
    model_key = "clinical_routing_segmentation"
    model_name = "Clinical segmentation router"

    def __init__(self, primary_engine: Optional[BaseSegmentationEngine], da_engine: Optional[DaNnUnetSegmentationEngine] = None):
        self.primary_engine = primary_engine
        self.da_engine = da_engine

    def is_ready(self) -> bool:
        return bool((self.da_engine and self.da_engine.is_ready()) or (self.primary_engine and self.primary_engine.is_ready()))

    def segment(
        self,
        volume: np.ndarray,
        voxel_spacing: Optional[Sequence[float]] = None,
        series_labels: Optional[Sequence[str]] = None,
    ) -> SegmentationPrediction:
        if self.da_engine is not None and self.da_engine.is_ready() and self.da_engine.can_handle(volume, series_labels):
            try:
                prediction = self.da_engine.segment(volume, voxel_spacing=voxel_spacing, series_labels=series_labels)
                prediction.metadata["routing_selected_model"] = self.da_engine.model_key
                prediction.metadata["routing_fallback_available"] = bool(self.primary_engine is not None)
                return prediction
            except Exception as exc:
                if self.primary_engine is None:
                    raise
                fallback_prediction = self.primary_engine.segment(volume, voxel_spacing=voxel_spacing, series_labels=series_labels)
                fallback_prediction.metadata["routing_selected_model"] = getattr(self.primary_engine, "model_key", "primary")
                fallback_prediction.metadata["routing_da_error"] = str(exc)
                return fallback_prediction

        if self.primary_engine is None:
            raise RuntimeError("No available fallback segmentation engine")

        prediction = self.primary_engine.segment(volume, voxel_spacing=voxel_spacing, series_labels=series_labels)
        prediction.metadata["routing_selected_model"] = getattr(self.primary_engine, "model_key", "primary")
        prediction.metadata["routing_da_available"] = bool(self.da_engine is not None and self.da_engine.is_ready())
        return prediction


def build_best_segmentation_engine(model_path: Path, model_key: str, device: str = "auto") -> Optional[BaseSegmentationEngine]:
    if model_key == DaNnUnetSegmentationEngine.model_key:
        return DaNnUnetSegmentationEngine(model_dir=model_path, device=device)

    primary: Optional[BaseSegmentationEngine] = None
    if model_key == MonaiBraTSSegmentationEngine.model_key:
        primary = MonaiBraTSSegmentationEngine(model_path=model_path, device=device)
        secondary_path = model_path.parents[2] / NnUnet3DStrongEngine.model_key / "best.pt"
        if secondary_path.exists():
            try:
                secondary = NnUnet3DStrongEngine(model_path=secondary_path, device=device)
                primary = WeightedEnsembleSegmentationEngine([primary, secondary])
            except Exception:
                pass
    elif model_key == NnUnet3DStrongEngine.model_key:
        primary = NnUnet3DStrongEngine(model_path=model_path, device=device)

    da_engine: Optional[DaNnUnetSegmentationEngine] = None
    da_model_dir = Path(os.getenv("DA_NNUNET_MODEL_DIR", str(DEFAULT_DA_NNUNET_MODEL_DIR)))
    if da_model_dir.exists():
        try:
            da_engine = DaNnUnetSegmentationEngine(model_dir=da_model_dir, device=device)
        except Exception:
            da_engine = None

    if primary is not None and da_engine is not None:
        return ClinicalRoutingSegmentationEngine(primary_engine=primary, da_engine=da_engine)
    return da_engine or primary
