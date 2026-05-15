# Import các thư viện cần thiết
import os  # Để truy cập biến môi trường
from pathlib import Path  # Để xử lý đường dẫn file

from dotenv import load_dotenv  # Để tải các biến từ file .env

load_dotenv()  # Tải các biến môi trường từ file .env

# Xác định thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent

# Cấu hình tên dự án và phiên bản
PROJECT_NAME = os.getenv("PROJECT_NAME", "TumorSight - AI 3D Tumor Detection & Analysis Platform")  # Tên ứng dụng
PROJECT_VERSION = "1.0.0"  # Phiên bản hiện tại
PROJECT_DESCRIPTION = "AI 3D Tumor Intelligence & Teleconsultation Platform"  # Mô tả dự án

# Cấu hình API
API_PREFIX = os.getenv("API_PREFIX", "/api")  # Tiền tố đường dẫn API
HOST = os.getenv("HOST", "127.0.0.1")  # Địa chỉ host của server
PORT = int(os.getenv("PORT", "8000"))  # Cổng chạy server
DEBUG = os.getenv("DEBUG", "False").lower() == "true"  # Chế độ debug (phát triển hay sản xuất)

# Cấu hình đường dẫn thư mục
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static")))  # Thư mục chứa file tĩnh (HTML, CSS, JS)
STATIC_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

TMP_UPLOAD_DIR = Path(os.getenv("TMP_UPLOAD_DIR", str(BASE_DIR / "tmp")))  # Thư mục tạm để lưu file upload
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục tạm

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))  # Thư mục chứa dữ liệu
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục dữ liệu

CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", str(BASE_DIR / "checkpoints")))  # Thư mục chứa các checkpoint của model
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục checkpoint
BENCHMARK_DIR = Path(os.getenv("BENCHMARK_DIR", str(CHECKPOINTS_DIR / "benchmark")))  # Thư mục benchmark
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục benchmark

CASE_ARTIFACTS_DIR = Path(os.getenv("CASE_ARTIFACTS_DIR", str(TMP_UPLOAD_DIR / "cases")))  # Thư mục lưu kết quả phân tích từng trường hợp
CASE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục artifacts


# Cấu hình đường dẫn của các model nhân dạng u não
MODEL_PATHS = {
    # Model MONAI BraTS - sử dụng SegResNet
    "monai_brats_mri_segmentation": CHECKPOINTS_DIR / "monai_brats_mri_segmentation" / "models" / "model.pt",
    # nnU-Net 3D Strong v3 - phiên bản mạnh nhất v3
    "nnunet3d_strong_v3": CHECKPOINTS_DIR / "nnunet3d_strong_v3" / "best.pt",
    # nnU-Net 3D Strong - phiên bản đã được huấn luyện
    "nnunet3d_strong": CHECKPOINTS_DIR / "nnunet3d_strong" / "best.pt",
    # RA nnU-Net 3D ASPP-SE v2 - với attention mechanism
    "ra_nnunet3d_ds_aspp_se_v2": CHECKPOINTS_DIR / "ra_nnunet3d_ds_aspp_se_v2" / "best.pt",
    # UNet3D CPU Light - phiên bản nhẹ tối ưu cho CPU
    "unet3d_cpu_light": CHECKPOINTS_DIR / "unet3d_cpu_light" / "best.pt",
}

# Ánh xạ tên hiển thị cho các model
MODEL_NAMES = {
    "monai_brats_mri_segmentation": "MONAI BraTS SegResNet Bundle",  # Tên hiển thị cho MONAI model
    "nnunet3d_strong_v3": "nnU-Net 3D Strong v3",  # Tên hiển thị cho nnU-Net v3
    "nnunet3d_strong": "nnU-Net 3D Strong",  # Tên hiển thị cho nnU-Net
    "ra_nnunet3d_ds_aspp_se_v2": "RA nnU-Net 3D ASPP-SE v2",  # Tên hiển thị cho RA model
    "unet3d_cpu_light": "UNet3D CPU Light",  # Tên hiển thị cho UNet CPU light
}

# Cấu hình model DA-nnUNet cho dữ liệu nhi đồng
DA_NNUNET_MODEL_KEY = "da_nnunet_pediatric"  # Khóa định danh cho model DA-nnUNet
DA_NNUNET_SOURCE_DIR = Path(
    os.getenv(
        "DA_NNUNET_SOURCE_DIR",
        str(BASE_DIR / "external" / "DA_nnUNet"),
    )
)  # Đường dẫn source code DA-nnUNet trong external

# Tên thư mục con chứa cấu hình DA-nnUNet
_DA_NNUNET_DATASET_DIR = "Dataset142_BraTS2023_MIXED_GRL"
_DA_NNUNET_TRAINER_DIR = "nnUNetTrainerDA_500ep_noDS_4Convs__nnUNetPlans__3d_fullres_bs4"


def _resolve_da_nnunet_model_dir() -> Path:
    """Tìm thư mục weights DA-nnUNet từ biến môi trường hoặc các vị trí mặc định.

    Ưu tiên: (1) biến môi trường DA_NNUNET_MODEL_DIR,
    (2) checkpoints/da_nnunet/DA_nnUNet/...,
    (3) external/DA_nnUNet/nnUNet_results/...
    """
    env_override = os.getenv("DA_NNUNET_MODEL_DIR")
    if env_override:
        return Path(env_override)

    # Tìm trong checkpoints/da_nnunet/ trước (vị trí thực tế trên nhiều máy)
    checkpoints_candidate = (
        CHECKPOINTS_DIR / "da_nnunet" / "DA_nnUNet"
        / _DA_NNUNET_DATASET_DIR / _DA_NNUNET_TRAINER_DIR
    )
    if checkpoints_candidate.exists():
        return checkpoints_candidate

    # Fallback: tìm trong external/DA_nnUNet/nnUNet_results/ (cấu trúc gốc)
    external_candidate = (
        DA_NNUNET_SOURCE_DIR / "nnUNet_results"
        / _DA_NNUNET_DATASET_DIR / _DA_NNUNET_TRAINER_DIR
    )
    if external_candidate.exists():
        return external_candidate

    # Trả về đường dẫn checkpoints (cho thông báo lỗi rõ ràng hơn)
    return checkpoints_candidate


DA_NNUNET_MODEL_DIR = _resolve_da_nnunet_model_dir()  # Đường dẫn thư mục chứa weights của model DA-nnUNet
MODEL_NAMES[DA_NNUNET_MODEL_KEY] = "DA-nnUNet Pediatric Domain-Adapted"  # Tên hiển thị cho model DA-nnUNet


def get_model_artifact_path(model_key: str) -> Path:
    """Lấy đường dẫn tới file checkpoint của model theo khóa"""
    return MODEL_PATHS.get(model_key, CHECKPOINTS_DIR / model_key / "best.pt")  # Trả về đường dẫn hoặc tạo đường dẫn mặc định


def _resolve_existing_checkpoint() -> Path:
    """Tìm checkpoint model hiện tại từ các vị trí cấu hình"""
    configured = os.getenv("SEGMENTATION_CHECKPOINT_PATH")  # Lấy đường dẫn từ biến môi trường
    candidates = []  # Danh sách các đường dẫn ứng viên
    if configured:
        configured_path = Path(configured)  # Chuyển đổi chuỗi thành Path
        if not configured_path.is_absolute():  # Nếu là đường dẫn tương đối
            configured_path = BASE_DIR / configured_path  # Chuyển thành đường dẫn tuyệt đối
        candidates.append(configured_path)  # Thêm vào danh sách ứng viên

    candidates.extend(get_model_artifact_path(model_key) for model_key in MODEL_PATHS)  # Thêm tất cả đường dẫn model

    for candidate in candidates:  # Duyệt qua từng ứng viên
        if candidate.exists():  # Nếu file tồn tại
            return candidate  # Trả về đường dẫn này

    return candidates[0]  # Trả về ứng viên đầu tiên nếu không tìm thấy


ACTIVE_CHECKPOINT_PATH = _resolve_existing_checkpoint()  # Tìm và lưu đường dẫn checkpoint hiện tại


def _resolve_model_key_from_path(path: Path) -> str:
    """Xác định khóa model từ đường dẫn file"""
    resolved = path.resolve(strict=False)  # Chuyển đổi thành đường dẫn tuyệt đối
    for model_key, model_path in MODEL_PATHS.items():  # Duyệt qua từng model trong từ điển
        if resolved == model_path.resolve(strict=False):  # Nếu đường dẫn khớp
            return model_key  # Trả về khóa model
    return path.parent.name  # Nếu không tìm thấy, trả về tên thư mục cha


# Cấu hình checkpoint model tốt nhất được chọn
BEST_CHECKPOINT_KEY = _resolve_model_key_from_path(ACTIVE_CHECKPOINT_PATH)  # Xác định loại model
BEST_CHECKPOINT_PATH = str(ACTIVE_CHECKPOINT_PATH)  # Đường dẫn tuyệt đối tới file checkpoint
BEST_CHECKPOINT_NAME = MODEL_NAMES.get(BEST_CHECKPOINT_KEY, ACTIVE_CHECKPOINT_PATH.parent.name)  # Tên hiển thị của model
BEST_CHECKPOINT_TYPE = "official_bundle" if BEST_CHECKPOINT_KEY == "monai_brats_mri_segmentation" else "custom_trained"  # Loại checkpoint (chính thức hay tự huấn luyện)
BEST_CHECKPOINT_DICE = 0.8518 if BEST_CHECKPOINT_KEY == "monai_brats_mri_segmentation" else 0.92  # Điểm DICE của model

# Cấu hình xử lý phân đoạn (Segmentation)
SEGMENTATION_INPUT_SHAPE = (1, 128, 128, 128)  # Kích thước đầu vào: (channels, height, width, depth)
SEGMENTATION_NUM_CLASSES = 2  # Số lớp: 0 = nền, 1 = u não
SEGMENTATION_DEVICE = os.getenv("SEGMENTATION_DEVICE", "auto")  # Thiết bị tính toán: GPU hoặc CPU

# Cấu hình xử lý ảnh
IMAGE_NORMALIZATION_MEAN = [0.5]  # Giá trị trung bình để chuẩn hóa ảnh
IMAGE_NORMALIZATION_STD = [0.5]  # Độ lệch chuẩn để chuẩn hóa ảnh
VOXEL_SPACING = (1.0, 1.0, 1.0)  # Kích thước voxel trong không gian (mm)

# Cấu hình cơ sở dữ liệu (tùy chọn SQLite)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tumorsight.db")  # Địa chỉ kết nối cơ sở dữ liệu

# Cấu hình ghi log
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Mức độ chi tiết của log
SEGMENTATION_TUNING_PATH = str(Path(os.getenv("SEGMENTATION_TUNING_PATH", str(BENCHMARK_DIR / "segmentation_tuning.json"))))  # Đường dẫn tới file cấu hình tuning


def get_checkpoint_path() -> str:
    """Trả về đường dẫn tới checkpoint model hiện tại, nếu không tồn tại sẽ ném lỗi"""
    checkpoint_path = Path(BEST_CHECKPOINT_PATH)  # Chuyển đổi chuỗi thành Path
    if checkpoint_path.exists():  # Kiểm tra file có tồn tại
        return str(checkpoint_path)  # Trả về đường dẫn dạng chuỗi
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")  # Ném lỗi nếu file không tồn tại
