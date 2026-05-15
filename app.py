from __future__ import annotations
import base64  # Để mã hóa/giải mã base64
from dataclasses import dataclass  # Để tạo class dữ liệu đơn giản
import json  # Để xử lý JSON
import logging  # Để ghi log
import re  # Để xử lý regular expression
import uuid  # Để tạo ID duy nhất
from datetime import date, datetime, timedelta  # Để xử lý ngày giờ
from hashlib import sha256  # Để tính hash SHA256
from pathlib import Path  # Để xử lý đường dẫn file
from typing import Any, Dict, List, Optional, Sequence, Tuple  # Các type hint

import numpy as np  # Thư viện toán học và xử lý mảng
from scipy import ndimage  # Các hàm xử lý ảnh từ SciPy
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # Framework web FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Middleware CORS cho phép yêu cầu từ các domain khác
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse  # Các loại response khác nhau
from fastapi.staticfiles import StaticFiles  # Để phục vụ file tĩnh (HTML, CSS, JS)

from config import (
    # Các cấu hình cơ bản
    API_PREFIX,  # Tiền tố của API endpoint
    BEST_CHECKPOINT_KEY,  # Khóa định danh của model tốt nhất
    BEST_CHECKPOINT_NAME,  # Tên hiển thị của model tốt nhất
    BEST_CHECKPOINT_PATH,  # Đường dẫn tới file checkpoint tốt nhất
    CASE_ARTIFACTS_DIR,  # Thư mục lưu trữ kết quả phân tích từng trường hợp
    CHECKPOINTS_DIR,  # Thư mục chứa tất cả checkpoint
    DATA_DIR,  # Thư mục chứa dữ liệu
    DA_NNUNET_MODEL_DIR,  # Thư mục model DA-nnUNet
    DA_NNUNET_MODEL_KEY,  # Khóa định danh DA-nnUNet
    DEBUG,  # Chế độ debug
    MODEL_NAMES,  # Ánh xạ tên hiển thị các model
    PROJECT_DESCRIPTION,  # Mô tả dự án
    PROJECT_NAME,  # Tên dự án
    PROJECT_VERSION,  # Phiên bản dự án
    STATIC_DIR,  # Thư mục chứa file tĩnh
    TMP_UPLOAD_DIR,  # Thư mục upload tạm thời
    VOXEL_SPACING,  # Kích thước voxel
    get_model_artifact_path,  # Hàm lấy đường dẫn artifact của model
)
# Import các module xử lý dữ liệu
from modules.classification import ClassificationEngine  # Engine phân loại u não
from modules.morphology import AdvancedMorphologyAnalyzer  # Phân tích hình thái u
from modules.progression import ProgressionAnalyzer  # Dự đoán tiến triển u
from modules.reconstruction import Reconstruction3D  # Tái dựng 3D u
from modules.segmentation import SegmentationPrediction, build_best_segmentation_engine  # Phân đoạn u
from schemas import (
    # Các lược đồ dữ liệu Pydantic
    AISummary,  # Tóm tắt AI
    AnalysisJob,  # Công việc phân tích
    CaseRecord,  # Bản ghi trường hợp
    ClassificationResult,  # Kết quả phân loại
    EvaluationEvidence,  # Bằng chứng đánh giá
    FollowUpPlan,  # Kế hoạch theo dõi
    IngestionInfo,  # Thông tin nhập liệu
    InsightSummary,  # Tóm tắt thông tin chi tiết
    MorphologySummary,  # Tóm tắt hình thái
    PatientInfo,  # Thông tin bệnh nhân
    PipelineStep,  # Bước trong pipeline
    PlatformEvidence,  # Bằng chứng nền tảng
    PlatformHealth,  # Tình trạng khỏe mạnh của nền tảng
    PlatformHealthCheckpoints,  # Checkpoint kiểm tra sức khỏe
    PlatformHealthChecks,  # Các kiểm tra sức khỏe
    PlatformInfo,  # Thông tin nền tảng
    PortfolioSummary,  # Tóm tắt danh mục
    PreprocessingInfo,  # Thông tin tiền xử lý
    ProgressionPrediction,  # Dự đoán tiến triển
    ReconstructionResult,  # Kết quả tái dựng
    ReconstructionViewerPayload,  # Dữ liệu viewer tái dựng
    SegmentationPreview,  # Xem trước phân đoạn
    SegmentationSlicePreview,  # Xem trước lát cắt phân đoạn
    SegmentationResult,  # Kết quả phân đoạn
    StudyInfo,  # Thông tin nghiên cứu
    TeleconsultationBoard,  # Bảng tư vấn từ xa
    TeleconsultationNote,  # Ghi chú tư vấn từ xa
    TeleconsultationStatus,  # Trạng thái tư vấn từ xa
    TimelineSnapshot,  # Ảnh chụp timeline
    TrainingEvidence,  # Bằng chứng huấn luyện
    TreatmentOption,  # Tùy chọn điều trị
    TreatmentSimulation,  # Mô phỏng điều trị
    ViewerOverlayPayload,  # Dữ liệu overlay viewer
    ViewerSliceOverlay,  # Overlay lát cắt viewer
    FollowUpReminder,  # Nhắc nhở theo dõi
)

logging.basicConfig(level=logging.INFO)  # Cấu hình ghi log ở mức INFO
logger = logging.getLogger(__name__)  # Tạo logger cho module này

NO_CACHE_HEADERS = {
    # Các header HTTP để không lưu cache file tĩnh
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",  # Không lưu cache
    "Pragma": "no-cache",  # Không lưu cache (tương thích cũ)
    "Expires": "0",  # Hết hạn ngay
}

MAX_UPLOAD_FILE_COUNT = 256
MAX_UPLOAD_FILE_SIZE_MB = 128
DEFAULT_DEIDENTIFIED_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "InstitutionName",
]


class NoCacheStaticFiles(StaticFiles):
    """Class tùy chỉnh để phục vụ file tĩnh mà không lưu cache"""
    async def get_response(self, path: str, scope: Dict[str, Any]):
        response = await super().get_response(path, scope)  # Lấy response từ class cha
        if getattr(response, "status_code", 200) == 200:  # Nếu response thành công
            response.headers.update(NO_CACHE_HEADERS)  # Thêm headers không cache
        return response

app = FastAPI(
    # Tạo ứng dụng FastAPI với thông tin dự án
    title=PROJECT_NAME,  # Tên dự án
    description=PROJECT_DESCRIPTION,  # Mô tả dự án
    version=PROJECT_VERSION,  # Phiên bản dự án
)

app.add_middleware(
    # Thêm middleware CORS để cho phép yêu cầu từ các domain khác
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin
    allow_credentials=True,  # Cho phép gửi credentials
    allow_methods=["*"],  # Cho phép tất cả HTTP methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

if STATIC_DIR.exists():
    # Nếu thư mục file tĩnh tồn tại
    app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")  # Mount file tĩnh


CASE_STORE: Dict[str, Dict[str, Any]] = {}  # Kho lưu trữ dữ liệu trường hợp bệnh nhân
ANALYSIS_JOBS: Dict[str, Dict[str, Any]] = {}  # Kho lưu trữ các công việc phân tích
ARRAY_CACHE: Dict[str, Dict[str, np.ndarray]] = {}  # Cache các mảng dữ liệu numpy
SEGMENTATION_RESULT_CACHE: Dict[str, SegmentationPrediction] = {}  # Cache segmentation theo fingerprint volume
CLASSIFIER: Optional[ClassificationEngine] = None  # Engine phân loại sẽ được tải lúc runtime
SEGMENTATION_ENGINE = None  # Engine phân đoạn sẽ được tải lúc runtime

CASE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục lưu kết quả phân tích
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Tạo thư mục upload tạm


@dataclass
class StudyVolume:
    """Lớp dữ liệu để lưu trữ thông tin khối lượng hình ảnh y tế"""
    volume: np.ndarray  # Mảng 3D chứa dữ liệu voxel
    voxel_spacing_mm: Tuple[float, float, float]  # Kích thước voxel trong từng chiều (mm)
    source_type: str  # Loại nguồn dữ liệu (DICOM, NIfTI, v.v.)
    series_labels: Tuple[str, ...] = ()  # Nhãn của các chuỗi ảnh (T1, T2, FLAIR, v.v.)


def load_runtime_models(force: bool = False) -> None:
    """Tải các model ML vào bộ nhớ khi server khởi động"""
    global CLASSIFIER, SEGMENTATION_ENGINE

    if force or CLASSIFIER is None:  # Nếu cần load lại hoặc chưa load
        try:
            CLASSIFIER = ClassificationEngine()  # Tạo instance engine phân loại
        except Exception as exc:
            logger.warning("Classifier engine is unavailable: %s", exc)  # Ghi warning nếu load thất bại
            CLASSIFIER = None

    if force or SEGMENTATION_ENGINE is None:  # Nếu cần load lại hoặc chưa load
        try:
            SEGMENTATION_ENGINE = build_best_segmentation_engine(Path(BEST_CHECKPOINT_PATH), BEST_CHECKPOINT_KEY)  # Xây dựng engine phân đoạn
        except Exception as exc:
            logger.warning("Primary segmentation engine is unavailable: %s", exc)  # Ghi warning nếu load thất bại
            SEGMENTATION_ENGINE = None


def classifier_engine() -> ClassificationEngine:
    """Lấy engine phân loại, load nếu chưa có"""
    load_runtime_models()  # Đảm bảo các model được load
    if CLASSIFIER is None:  # Nếu vẫn None
        raise RuntimeError("Khong the khoi tao mo hinh phan loai")  # Ném lỗi
    return CLASSIFIER


def runtime_model_registry() -> List[Dict[str, Any]]:
    """Tạo danh sách các model có sẵn và tình trạng của chúng"""
    load_runtime_models()  # Đảm bảo các model được load
    segmentation_path = Path(BEST_CHECKPOINT_PATH)  # Đường dẫn tới model phân đoạn
    da_model_dir = Path(DA_NNUNET_MODEL_DIR)  # Đường dẫn tới model DA-nnUNet
    classifier_checkpoint = getattr(CLASSIFIER, "checkpoint_path", None) if CLASSIFIER is not None else None  # Lấy đường dẫn checkpoint phân loại
    classifier_mode = getattr(CLASSIFIER, "runtime_mode", "unavailable") if CLASSIFIER is not None else "unavailable"  # Chế độ chạy classifier
    classifier_name = getattr(CLASSIFIER, "model_name", "Classification engine unavailable") if CLASSIFIER is not None else "Classification engine unavailable"  # Tên classifier
    segmentation_name = getattr(SEGMENTATION_ENGINE, "model_name", BEST_CHECKPOINT_NAME) if SEGMENTATION_ENGINE is not None else BEST_CHECKPOINT_NAME  # Tên model phân đoạn
    segmentation_mode = SEGMENTATION_ENGINE.__class__.__name__ if SEGMENTATION_ENGINE is not None else "unavailable"  # Chế độ chạy segmentation

    models = [
        {
            "task": "segmentation",  # Nhiệm vụ phân đoạn
            "key": BEST_CHECKPOINT_KEY,  # Khóa model
            "name": segmentation_name,  # Tên hiển thị
            "runtime_mode": segmentation_mode,  # Loại engine
            "ready": bool(SEGMENTATION_ENGINE is not None and segmentation_path.exists()),  # Có sẵn hay không
            "checkpoint_path": str(segmentation_path),  # Đường dẫn file
            "preferred": True,  # Là model ưa thích
        },
        {
            "task": "classification",  # Nhiệm vụ phân loại
            "key": "classification_3d",  # Khóa model
            "name": classifier_name,  # Tên hiển thị
            "runtime_mode": classifier_mode,  # Loại engine
            "ready": bool(CLASSIFIER is not None),  # Có sẵn hay không
            "checkpoint_path": str(classifier_checkpoint) if classifier_checkpoint else None,  # Đường dẫn file
            "preferred": True,  # Là model ưa thích
        },
    ]
    if da_model_dir.exists():  # Nếu model DA-nnUNet tồn tại
        models.append(
            {
                "task": "segmentation",  # Nhiệm vụ phân đoạn
                "key": DA_NNUNET_MODEL_KEY,  # Khóa model DA-nnUNet
                "name": MODEL_NAMES.get(DA_NNUNET_MODEL_KEY, "DA-nnUNet Pediatric Domain-Adapted"),  # Tên hiển thị
                "runtime_mode": "DaNnUnetSegmentationEngine",  # Loại engine
                "ready": True,  # Sẵn sàng
                "checkpoint_path": str(da_model_dir),  # Đường dẫn thư mục
                "preferred": True,  # Là model ưa thích
            }
        )
    return models  # Trả về danh sách model


def now_iso() -> str:
    """Lấy thời gian hiện tại theo định dạng ISO 8601"""
    return datetime.now().replace(microsecond=0).isoformat()  # Loại bỏ microsecond


def today_iso() -> str:
    """Lấy ngày hôm nay theo định dạng ISO 8601"""
    return date.today().isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    """Giới hạn giá trị trong khoảng [lower, upper]"""
    return max(lower, min(upper, value))  # Đảm bảo value nằm trong [lower, upper]


def as_float(value: Any, digits: int = 4) -> float:
    """Chuyển đổi giá trị thành float và làm tròn đến số chữ số thập phân"""
    return round(float(value), digits)  # Làm tròn đến số chữ số chỉ định


def nhan_phan_loai(label: str) -> str:
    """Chuyển đổi nhãn phân loại từ tiếng Anh sang tiếng Việt"""
    return {
        "benign": "lành tính",  # Lành tính
        "malignant": "ác tính",  # Ác tính
        "uncertain": "không xác định",  # Không xác định
        "indeterminate": "không xác định",  # Không xác định
    }.get(label, label)  # Trả về giá trị dịch hoặc label gốc


def patient_code_from_name(patient_name: str) -> str:
    """Tạo mã bệnh nhân duy nhất từ tên"""
    digest = sha256(patient_name.strip().lower().encode("utf-8")).hexdigest()[:8].upper()  # Tính hash SHA256 của tên, lấy 8 ký tự đầu
    return f"BN-{digest}"  # Trả về mã dạng BN-xxxxxxxx


def to_serializable(value: Any) -> Any:
    """Chuyển các kiểu dữ liệu đặc biệt thành dạng serializable cho JSON"""
    if isinstance(value, np.ndarray):  # Nếu là mảng numpy
        return value.tolist()  # Chuyển thành list Python
    if isinstance(value, np.floating):  # Nếu là số thực numpy
        return float(value)  # Chuyển thành float Python
    if isinstance(value, np.integer):  # Nếu là số nguyên numpy
        return int(value)  # Chuyển thành int Python
    if isinstance(value, Path):  # Nếu là Path object
        return str(value)  # Chuyển thành chuỗi
    if isinstance(value, (datetime, date)):  # Nếu là ngày/giờ
        return value.isoformat()  # Chuyển thành ISO format
    return value  # Trả về nguyên vẹn nếu không cần chuyển


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Ghi dữ liệu vào file JSON"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=to_serializable),  # Chuyển thành JSON string
        encoding="utf-8",  # Mã hóa UTF-8 để hỗ trợ tiếng Việt
    )


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Đọc dữ liệu từ file JSON"""
    if not path.exists():  # Nếu file không tồn tại
        return default or {}  # Trả về dict mặc định hoặc rỗng
    return json.loads(path.read_text(encoding="utf-8"))  # Đọc và parse JSON


def case_dir(case_id: str) -> Path:
    """Lấy thư mục lưu trữ kết quả phân tích của một trường hợp"""
    directory = CASE_ARTIFACTS_DIR / case_id  # Tạo đường dẫn thư mục
    directory.mkdir(parents=True, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    return directory


def arrays_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file lưu mảng numpy của trường hợp"""
    return case_dir(case_id) / "arrays.npz"  # File compressed numpy arrays


def bundle_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file JSON chứa dữ liệu của trường hợp"""
    return case_dir(case_id) / "bundle.json"  # File JSON dữ liệu


def report_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file báo cáo markdown"""
    return case_dir(case_id) / "report.md"  # File báo cáo


def preview_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file xem trước"""
    return case_dir(case_id) / "preview.json"  # File xem trước


def mask_artifact_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file mặt nạ phân đoạn"""
    return case_dir(case_id) / "mask.json"  # File mặt nạ


def probability_artifact_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file bản đồ xác suất"""
    return case_dir(case_id) / "probability-map.json"  # File bản đồ xác suất


def viewer_overlay_artifact_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file overlay viewer"""
    return case_dir(case_id) / "viewer-overlays.json"  # File overlay


def mpr_artifact_path(case_id: str) -> Path:
    """Lấy đường dẫn tới file MPR (Multiplanar Reconstruction)"""
    return case_dir(case_id) / "mpr.json"  # File MPR


def get_case_or_404(case_id: str) -> Dict[str, Any]:
    case = CASE_STORE.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Khong tim thay ca benh")
    return case


def load_case_arrays(case_id: str) -> Dict[str, np.ndarray]:
    cached = ARRAY_CACHE.get(case_id)
    if cached is not None:
        return cached

    path = arrays_path(case_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Khong tim thay du lieu mang cua ca benh")

    data = np.load(path)
    arrays = {name: data[name] for name in data.files}
    ARRAY_CACHE[case_id] = arrays
    return arrays


def parse_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON artifact: %s", path)
        return {}


def detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def available_model_keys() -> List[str]:
    keys = set(MODEL_NAMES)
    keys.update(directory.name for directory in CHECKPOINTS_DIR.iterdir() if directory.is_dir())
    return sorted(keys)


def model_name_from_key(key: str) -> str:
    return MODEL_NAMES.get(key, key.replace("_", " ").title())


def checkpoint_health() -> PlatformHealthCheckpoints:
    checkpoint_path = Path(BEST_CHECKPOINT_PATH)
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else None
    return PlatformHealthCheckpoints(
        preferred_model_key=BEST_CHECKPOINT_KEY,
        preferred_model_name=BEST_CHECKPOINT_NAME,
        preferred_model_path=str(checkpoint_path),
        training_checkpoint_ready=checkpoint_path.exists(),
        training_checkpoint_size_mb=as_float(size_mb, 2) if size_mb is not None else None,
        available_models=available_model_keys(),
    )


def training_evidence() -> TrainingEvidence:
    summary = parse_optional_json(CHECKPOINTS_DIR / "unet3d_cpu_light" / "training_summary.json")
    return TrainingEvidence(
        best_val_dice=summary.get("best_val_dice"),
        epochs=summary.get("epochs"),
        train_samples=summary.get("train_samples"),
        val_samples=summary.get("val_samples"),
    )


def evaluation_evidence() -> EvaluationEvidence:
    summary = parse_optional_json(CHECKPOINTS_DIR / "benchmark" / "segmentation_eval_summary.json")
    if not summary:
        summary = parse_optional_json(CHECKPOINTS_DIR / "nnunet3d_strong_v3" / "volume_eval_summary.json")
    metrics = summary.get("metrics", {})
    clean = metrics.get("clean", {})
    raw = metrics.get("raw", {})
    return EvaluationEvidence(
        clean_mae_cm3=clean.get("mae_cm3"),
        clean_mape_percent=clean.get("mape_percent"),
        raw_mape_percent=raw.get("mape_percent"),
        validation_cases=summary.get("case_count"),
    )


def dataset_ready_case_count() -> int:
    manifest = parse_optional_json(DATA_DIR / "brats_peds_prepared" / "manifest.json")
    return int(manifest.get("ready_case_count", 0) or 0)


def portfolio_summary() -> PortfolioSummary:
    cases = list(CASE_STORE.values())
    if not cases:
        return PortfolioSummary(
            case_count=0,
            avg_completion_score=0.0,
            avg_priority_score=0.0,
            ready_for_board=0,
            needs_attention=0,
            top_case_ids=[],
        )

    completion_scores = [float(case["segmentation"]["case_quality_score"]) * 100 for case in cases]
    priority_scores = [
        float(case["classification"]["risk_score"]) + max(float(case["progression"]["volume_change_6m_percent"]), 0)
        for case in cases
    ]
    ready_for_board = sum(1 for case in cases if case["ai_summary"]["review_status"] == "Sẵn sàng hội chẩn")
    needs_attention = sum(1 for case in cases if case["ai_summary"]["triage_level"] in {"Cao", "Nguy kịch"})
    top_case_ids = [
        case["case_id"]
        for case in sorted(
            cases,
            key=lambda item: float(item["classification"]["risk_score"])
            + max(float(item["progression"]["volume_change_6m_percent"]), 0) * 1.3,
            reverse=True,
        )[:5]
    ]

    return PortfolioSummary(
        case_count=len(cases),
        avg_completion_score=as_float(np.mean(completion_scores), 1),
        avg_priority_score=as_float(np.mean(priority_scores), 1),
        ready_for_board=ready_for_board,
        needs_attention=needs_attention,
        top_case_ids=top_case_ids,
    )


def build_platform_payload() -> Dict[str, Any]:
    models = runtime_model_registry()
    checkpoint_checks = checkpoint_health()
    dataset_cases = dataset_ready_case_count()
    health_checks = PlatformHealthChecks(
        api="ok",
        storage="ok" if CASE_ARTIFACTS_DIR.exists() else "missing",
        dataset="ready" if dataset_cases > 0 else "missing",
        checkpoints=checkpoint_checks,
    )
    degraded = (
        health_checks.storage != "ok"
        or health_checks.dataset != "ready"
        or not checkpoint_checks.training_checkpoint_ready
    )
    health = PlatformHealth(
        status="degraded" if degraded else "healthy",
        generated_at=now_iso(),
        checks=health_checks,
    )
    evidence = PlatformEvidence(training=training_evidence(), evaluation=evaluation_evidence())
    platform = PlatformInfo(
        name=PROJECT_NAME,
        version=PROJECT_VERSION,
        availability=True,
        gpu=detect_gpu(),
        memory=f"{len(ARRAY_CACHE)} cached / {len(CASE_STORE)} active",
        case_count=len(CASE_STORE),
        health=health,
        evidence=evidence,
        portfolio=portfolio_summary(),
    )
    payload = platform.model_dump(mode="json")
    payload["runtime_models"] = models
    return payload


def infer_format_labels(file_names: Sequence[str]) -> List[str]:
    labels = set()
    for name in file_names:
        suffix = Path(name).suffix.lower()
        if suffix in {".nii", ".gz", ".nii.gz"}:
            labels.add("NIfTI")
        elif suffix in {".dcm", ""}:
            labels.add("DICOM")
        else:
            labels.add(suffix.lstrip(".").upper())
    return sorted(labels or {"DICOM"})


def choose_inference_mode(file_count: int, timepoint_count: int) -> str:
    if file_count >= 120 or timepoint_count >= 4:
        return "toi uu du lieu lon"
    if file_count >= 40 or timepoint_count >= 3:
        return "can bang"
    return "nhanh"


SUPPORTED_VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz")
BRATS_MODALITIES = ("t1c", "t1", "t2", "flair")


def normalize_brats_modality(label: str) -> Optional[str]:
    normalized = "".join(character for character in str(label).strip().lower() if character.isalnum())
    if not normalized:
        return None
    aliases = {
        "t1c": {"t1c", "t1ce", "t1gd", "contrast", "postcontrast", "postgad"},
        "t1": {"t1"},
        "t2": {"t2"},
        "flair": {"flair", "t2f", "fla", "t2flair"},
    }
    for modality, values in aliases.items():
        if normalized in values:
            return modality
    return None


def detect_sequence_label(text: str) -> Optional[str]:
    name = re.sub(r"[_\-]+", " ", text.lower())

    contrast_patterns = (
        r"\bt1\s*(ce|c)\b",
        r"\bt1ce\b",
        r"\bt1gd\b",
        r"\bt1 gad\b",
        r"\bt1 post\b",
        r"\bpost.?contrast\b",
        r"\bpost gad\b",
        r"\bcontrast\b",
        r"\bc\+\b",
        r"\bgad\b",
        r"\bce\b",
    )
    if any(re.search(pattern, name) for pattern in contrast_patterns):
        return "t1c"
    if "flair" in name:
        return "flair"
    if re.search(r"(^|[^a-z0-9])t2([^a-z0-9]|$)", name):
        return "t2"
    if re.search(r"(^|[^a-z0-9])t1([^a-z0-9]|$)", name) or re.search(r"\b(mprage|spgr|bravo)\b", name):
        return "t1"
    return None


def _safe_dataset_text(dataset: Any, field_name: str) -> str:
    value = getattr(dataset, field_name, "")
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def detect_sequence_label_from_dataset(dataset: Any, fallback_name: str = "") -> Optional[str]:
    hints = " ".join(
        part
        for part in [
            fallback_name,
            _safe_dataset_text(dataset, "SeriesDescription"),
            _safe_dataset_text(dataset, "ProtocolName"),
            _safe_dataset_text(dataset, "SequenceName"),
            _safe_dataset_text(dataset, "SequenceVariant"),
            _safe_dataset_text(dataset, "ScanningSequence"),
            _safe_dataset_text(dataset, "ImageType"),
        ]
        if part
    )
    label = detect_sequence_label(hints)
    if label:
        return label

    contrast_agent = _safe_dataset_text(dataset, "ContrastBolusAgent").strip().lower()
    if contrast_agent:
        return "t1c"
    return None


def is_volume_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(extension) for extension in SUPPORTED_VOLUME_EXTENSIONS)


def _default_spacing() -> Tuple[float, float, float]:
    return tuple(float(item) for item in VOXEL_SPACING)


def _spacing_from_sitk(image) -> Tuple[float, float, float]:
    spacing_xyz = tuple(float(item) for item in image.GetSpacing())
    return tuple(reversed(spacing_xyz))


def read_volume_file(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    lower_name = path.name.lower()
    if lower_name.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float32), _default_spacing()
    if lower_name.endswith(".npz"):
        archive = np.load(path)
        payload_keys = [key for key in archive.files if key != "spacing"]
        if not payload_keys:
            raise ValueError(f"Archive {path} does not contain arrays")
        spacing = tuple(float(item) for item in archive["spacing"]) if "spacing" in archive.files else _default_spacing()
        return np.asarray(archive[payload_keys[0]], dtype=np.float32), spacing

    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.float32), _spacing_from_sitk(image)


def _dicom_series_uid(dataset: Any, fallback: Path) -> str:
    return str(getattr(dataset, "SeriesInstanceUID", fallback.parent.name or fallback.stem))


def _dicom_sort_position(dataset: Any, fallback_index: int) -> float:
    try:
        if hasattr(dataset, "ImagePositionPatient"):
            position = np.asarray([float(item) for item in dataset.ImagePositionPatient], dtype=np.float32)
            if hasattr(dataset, "ImageOrientationPatient"):
                orientation = np.asarray([float(item) for item in dataset.ImageOrientationPatient], dtype=np.float32)
                if orientation.size == 6:
                    normal = np.cross(orientation[:3], orientation[3:])
                    norm = float(np.linalg.norm(normal))
                    if norm > 1e-6:
                        return float(np.dot(position, normal / norm))
            return float(position[-1])
    except Exception:
        pass

    if hasattr(dataset, "SliceLocation"):
        try:
            return float(dataset.SliceLocation)
        except Exception:
            pass

    try:
        return float(getattr(dataset, "InstanceNumber"))
    except Exception:
        return float(fallback_index)


def _dicom_slice_spacing(datasets: Sequence[Any]) -> float:
    positions = []
    for dataset in datasets:
        try:
            if hasattr(dataset, "ImagePositionPatient"):
                position = np.asarray([float(item) for item in dataset.ImagePositionPatient], dtype=np.float32)
                if hasattr(dataset, "ImageOrientationPatient"):
                    orientation = np.asarray([float(item) for item in dataset.ImageOrientationPatient], dtype=np.float32)
                    if orientation.size == 6:
                        normal = np.cross(orientation[:3], orientation[3:])
                        norm = float(np.linalg.norm(normal))
                        if norm > 1e-6:
                            positions.append(float(np.dot(position, normal / norm)))
                            continue
                positions.append(float(position[-1]))
        except Exception:
            continue

    if len(positions) >= 2:
        ordered = np.unique(np.round(np.sort(np.asarray(positions, dtype=np.float32)), 5))
        diffs = np.diff(ordered)
        diffs = diffs[np.abs(diffs) > 1e-5]
        if diffs.size:
            return float(np.median(np.abs(diffs)))

    for dataset in datasets:
        try:
            thickness = float(getattr(dataset, "SpacingBetweenSlices"))
            if thickness > 0:
                return thickness
        except Exception:
            pass
        try:
            thickness = float(getattr(dataset, "SliceThickness"))
            if thickness > 0:
                return thickness
        except Exception:
            pass

    return float(VOXEL_SPACING[0])


def _pixel_spacing_from_dataset(dataset: Any) -> Tuple[float, float]:
    try:
        spacing = tuple(float(item) for item in dataset.PixelSpacing)
        if len(spacing) >= 2:
            return float(spacing[0]), float(spacing[1])
    except Exception:
        pass
    return float(VOXEL_SPACING[1]), float(VOXEL_SPACING[2])


def _dicom_series_candidate(label: str, files: Sequence[Path], datasets: Sequence[Any]) -> Optional[Dict[str, Any]]:
    if not datasets:
        return None

    ordered = []
    for index, dataset in enumerate(datasets):
        if not hasattr(dataset, "PixelData"):
            continue
        try:
            pixels = dataset.pixel_array.astype(np.float32)
            slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
            pixels = (pixels * slope) + intercept
            ordered.append((_dicom_sort_position(dataset, index), pixels, dataset))
        except Exception:
            continue

    if not ordered:
        return None

    ordered.sort(key=lambda item: item[0])
    sample = ordered[0][2]
    spacing_rc = _pixel_spacing_from_dataset(sample)
    spacing = (
        _dicom_slice_spacing([item[2] for item in ordered]),
        spacing_rc[0],
        spacing_rc[1],
    )
    description = " | ".join(
        part
        for part in [
            _safe_dataset_text(sample, "SeriesDescription"),
            _safe_dataset_text(sample, "ProtocolName"),
        ]
        if part
    )
    return {
        "label": label,
        "volume": np.stack([item[1] for item in ordered], axis=0).astype(np.float32),
        "spacing": tuple(float(item) for item in spacing),
        "slice_count": len(ordered),
        "description": description,
        "score": len(ordered),
        "files": [str(path) for path in files],
    }


def _select_labeled_dicom_candidates(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_label.setdefault(candidate["label"], []).append(candidate)

    selected: Dict[str, Dict[str, Any]] = {}
    for label, options in by_label.items():
        options = sorted(
            options,
            key=lambda item: (item["score"], np.prod(item["volume"].shape)),
            reverse=True,
        )
        selected[label] = options[0]
    return selected


def read_dicom_series(directory: Path) -> Optional[StudyVolume]:
    try:
        import pydicom
    except Exception:
        pydicom = None

    dicom_files = [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in {".dcm", ""}]
    if not dicom_files:
        return None

    if pydicom is not None:
        grouped_datasets: Dict[str, Dict[str, Any]] = {}
        for path in dicom_files:
            try:
                dataset = pydicom.dcmread(str(path), force=True)
            except Exception:
                continue
            if not hasattr(dataset, "PixelData"):
                continue
            series_uid = _dicom_series_uid(dataset, path)
            group = grouped_datasets.setdefault(series_uid, {"files": [], "datasets": []})
            group["files"].append(path)
            group["datasets"].append(dataset)

        candidates: List[Dict[str, Any]] = []
        unlabeled_candidates: List[Dict[str, Any]] = []
        for group in grouped_datasets.values():
            datasets = group["datasets"]
            sample = datasets[0]
            label = detect_sequence_label_from_dataset(sample, group["files"][0].name)
            candidate = _dicom_series_candidate(label or "unknown", group["files"], datasets)
            if not candidate:
                continue
            if label:
                candidates.append(candidate)
            else:
                unlabeled_candidates.append(candidate)

        selected = _select_labeled_dicom_candidates(candidates)
        if all(label in selected for label in BRATS_MODALITIES):
            reference = selected[BRATS_MODALITIES[0]]
            signature = (reference["volume"].shape, tuple(round(item, 4) for item in reference["spacing"]))
            if all(
                (selected[label]["volume"].shape, tuple(round(item, 4) for item in selected[label]["spacing"])) == signature
                for label in BRATS_MODALITIES
            ):
                return StudyVolume(
                    volume=np.stack([selected[label]["volume"] for label in BRATS_MODALITIES], axis=0).astype(np.float32),
                    voxel_spacing_mm=reference["spacing"],
                    source_type="uploaded_dicom",
                    series_labels=tuple(BRATS_MODALITIES),
                )

        if selected:
            ordered_labels = [label for label in BRATS_MODALITIES if label in selected]
            reference = selected[ordered_labels[0]]
            reference_shape = reference["volume"].shape
            aligned_labels = [label for label in ordered_labels if selected[label]["volume"].shape == reference_shape]
            if aligned_labels:
                stacked = np.stack([selected[label]["volume"] for label in aligned_labels], axis=0).astype(np.float32)
                if stacked.shape[0] == 1:
                    stacked = stacked[0]
                return StudyVolume(
                    volume=stacked,
                    voxel_spacing_mm=selected[aligned_labels[0]]["spacing"],
                    source_type="uploaded_dicom",
                    series_labels=tuple(aligned_labels),
                )

        if unlabeled_candidates:
            best = sorted(
                unlabeled_candidates,
                key=lambda item: (item["score"], np.prod(item["volume"].shape)),
                reverse=True,
            )[0]
            return StudyVolume(
                volume=best["volume"].astype(np.float32),
                voxel_spacing_mm=best["spacing"],
                source_type="uploaded_dicom",
                series_labels=(),
            )

    try:
        import SimpleITK as sitk

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
        if series_ids:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_names)
            image = reader.Execute()
            return StudyVolume(
                volume=sitk.GetArrayFromImage(image).astype(np.float32),
                voxel_spacing_mm=_spacing_from_sitk(image),
                source_type="uploaded_dicom",
                series_labels=(),
            )
    except Exception:
        pass

    return None


def load_uploaded_study(saved_paths: Sequence[Path]) -> Optional[StudyVolume]:
    if not saved_paths:
        return None

    modality_volumes: Dict[str, Tuple[np.ndarray, Tuple[float, float, float]]] = {}
    loose_volumes: List[Tuple[np.ndarray, Tuple[float, float, float]]] = []

    for path in saved_paths:
        if not is_volume_file(path):
            continue
        try:
            volume, spacing = read_volume_file(path)
        except Exception as exc:
            logger.warning("Skipping unreadable volume %s: %s", path.name, exc)
            continue

        label = detect_sequence_label(path.name)
        if label and label not in modality_volumes:
            modality_volumes[label] = (volume, spacing)
        else:
            loose_volumes.append((volume, spacing))

    if all(label in modality_volumes for label in BRATS_MODALITIES):
        reference_volume, reference_spacing = modality_volumes[BRATS_MODALITIES[0]]
        shape = reference_volume.shape
        if all(modality_volumes[label][0].shape == shape for label in BRATS_MODALITIES):
            return StudyVolume(
                volume=np.stack([modality_volumes[label][0] for label in BRATS_MODALITIES], axis=0).astype(np.float32),
                voxel_spacing_mm=reference_spacing,
                source_type="uploaded_volume",
                series_labels=tuple(BRATS_MODALITIES),
            )

    if modality_volumes:
        ordered_labels = [label for label in BRATS_MODALITIES if label in modality_volumes]
        reference_volume, reference_spacing = modality_volumes[ordered_labels[0]]
        shape = reference_volume.shape
        aligned_labels = [label for label in ordered_labels if modality_volumes[label][0].shape == shape]
        if aligned_labels:
            stacked = np.stack([modality_volumes[label][0] for label in aligned_labels], axis=0).astype(np.float32)
            if stacked.shape[0] == 1:
                stacked = stacked[0]
            return StudyVolume(
                volume=stacked,
                voxel_spacing_mm=reference_spacing,
                source_type="uploaded_volume",
                series_labels=tuple(aligned_labels),
            )

    if loose_volumes:
        reference_shape = loose_volumes[0][0].shape
        aligned = [(volume, spacing) for volume, spacing in loose_volumes if volume.shape == reference_shape]
        if len(aligned) >= 4:
            return StudyVolume(
                volume=np.stack([volume for volume, _ in aligned[:4]], axis=0).astype(np.float32),
                voxel_spacing_mm=aligned[0][1],
                source_type="uploaded_volume",
                series_labels=(),
            )
        return StudyVolume(
            volume=aligned[0][0].astype(np.float32),
            voxel_spacing_mm=aligned[0][1],
            source_type="uploaded_volume",
            series_labels=tuple(label for label in BRATS_MODALITIES if label in modality_volumes),
        )

    if any(path.suffix.lower() in {".dcm", ""} for path in saved_paths):
        return read_dicom_series(saved_paths[0].parent)

    return None


async def store_uploaded_files(upload_files: Sequence[UploadFile], target_dir: Path) -> List[Path]:
    if len(upload_files) > MAX_UPLOAD_FILE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"So luong tep vuot gioi han cho phep ({MAX_UPLOAD_FILE_COUNT})",
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for index, upload in enumerate(upload_files):
        safe_name = Path(upload.filename or f"study_{index}.bin").name
        destination = target_dir / safe_name
        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_FILE_SIZE_MB:
            await upload.close()
            raise HTTPException(
                status_code=400,
                detail=f"Tep '{safe_name}' vuot gioi han {MAX_UPLOAD_FILE_SIZE_MB}MB",
            )
        destination.write_bytes(content)
        saved_paths.append(destination)
        await upload.close()

    return saved_paths


def build_ingestion_safety_summary(saved_paths: Sequence[Path]) -> Dict[str, Any]:
    total_bytes = 0
    warnings: List[str] = []
    dicom_like = 0
    nifti_like = 0
    unknown = 0

    for path in saved_paths:
        try:
            total_bytes += int(path.stat().st_size)
        except OSError:
            warnings.append(f"Khong doc duoc kich thuoc tep: {path.name}")
            continue
        suffix = path.suffix.lower()
        if suffix in {".dcm", ""}:
            dicom_like += 1
        elif suffix in {".nii", ".gz"}:
            nifti_like += 1
        else:
            unknown += 1

    if unknown:
        warnings.append(f"Phat hien {unknown} tep co dinh dang khong ro, he thong da co gang tu dong doc.")
    if dicom_like and not nifti_like:
        warnings.append("Da ap dung de-identification metadata cho bo DICOM upload.")

    return {
        "total_upload_size_mb": as_float(total_bytes / (1024 * 1024), 3),
        "deidentification_applied": bool(dicom_like),
        "deidentified_tags": list(DEFAULT_DEIDENTIFIED_TAGS if dicom_like else []),
        "upload_warnings": warnings,
    }


def segmentation_cache_key(
    volume: np.ndarray,
    voxel_spacing: Optional[Sequence[float]],
    series_labels: Optional[Sequence[str]],
    engine_key: str,
    runtime_mode: str,
) -> str:
    image = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    spacing = tuple(float(item) for item in (voxel_spacing or VOXEL_SPACING))
    labels = tuple(str(item).lower() for item in (series_labels or ()))
    digest = sha256()
    digest.update(str(engine_key).encode("utf-8"))
    digest.update(str(runtime_mode).encode("utf-8"))
    digest.update(str(image.shape).encode("utf-8"))
    digest.update(str(spacing).encode("utf-8"))
    digest.update(str(labels).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def clone_prediction(prediction: SegmentationPrediction) -> SegmentationPrediction:
    return SegmentationPrediction(
        model_key=prediction.model_key,
        model_name=prediction.model_name,
        mask=prediction.mask.copy(),
        probability=prediction.probability.copy(),
        threshold=float(prediction.threshold),
        quality_score=float(prediction.quality_score),
        confidence_score=float(prediction.confidence_score),
        inference_mode=str(prediction.inference_mode),
        input_shape=tuple(int(dim) for dim in prediction.input_shape),
        metadata={**prediction.metadata},
    )


def run_segmentation_with_best_model(
    volume: np.ndarray,
    voxel_spacing: Optional[Sequence[float]] = None,
    series_labels: Optional[Sequence[str]] = None,
    runtime_mode: str = "full",
) -> SegmentationPrediction:
    load_runtime_models()
    if SEGMENTATION_ENGINE is None:
        raise RuntimeError("Khong the khoi tao mo hinh phan doan")
    engine_key = str(getattr(SEGMENTATION_ENGINE, "model_key", "segmentation"))
    key = segmentation_cache_key(
        volume=volume,
        voxel_spacing=voxel_spacing,
        series_labels=series_labels,
        engine_key=engine_key,
        runtime_mode=runtime_mode,
    )
    cached = SEGMENTATION_RESULT_CACHE.get(key)
    if cached is not None:
        prediction = clone_prediction(cached)
        prediction.metadata["cache_hit"] = True
        prediction.metadata["cache_key"] = key[:12]
        return prediction

    prediction = SEGMENTATION_ENGINE.segment(volume, voxel_spacing=voxel_spacing, series_labels=series_labels)
    prediction.metadata["cache_hit"] = False
    prediction.metadata["cache_key"] = key[:12]
    if len(SEGMENTATION_RESULT_CACHE) >= 10:
        oldest = next(iter(SEGMENTATION_RESULT_CACHE.keys()))
        SEGMENTATION_RESULT_CACHE.pop(oldest, None)
    SEGMENTATION_RESULT_CACHE[key] = clone_prediction(prediction)
    return prediction


def lesion_extent_mm(mask: np.ndarray, voxel_spacing: Sequence[float]) -> List[float]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return [0.0, 0.0, 0.0]
    extent = (coords.max(axis=0) - coords.min(axis=0) + 1) * np.asarray(voxel_spacing, dtype=np.float32)
    return [as_float(value, 2) for value in extent]


def build_viewer_payload(reconstruction: Dict[str, Any], mask: np.ndarray) -> ReconstructionViewerPayload:
    mesh = reconstruction.get("mesh", {})
    vertices = mesh.get("vertices", [])
    faces = mesh.get("faces", [])
    shell_vertices = mesh.get("shell_vertices", [])
    shell_faces = mesh.get("shell_faces", [])
    spacing = np.asarray(reconstruction.get("voxel_spacing_mm", VOXEL_SPACING), dtype=np.float32)
    payload_kwargs = {
        "bounds_min_mm": [as_float(value, 3) for value in reconstruction.get("bounds_min_mm", [])],
        "bounds_max_mm": [as_float(value, 3) for value in reconstruction.get("bounds_max_mm", [])],
        "centroid_mm": [as_float(value, 3) for value in reconstruction.get("centroid_mm", [])],
        "dimensions_mm": [as_float(value, 3) for value in reconstruction.get("dimensions_mm", [])],
        "voxel_spacing_mm": [as_float(value, 4) for value in reconstruction.get("voxel_spacing_mm", [])],
        "mesh_quality_score": as_float(reconstruction.get("mesh_quality_score", 0.0), 4),
        "vertex_count": int(reconstruction.get("vertex_count", len(vertices) or 0) or 0),
        "face_count": int(reconstruction.get("face_count", len(faces) or 0) or 0),
        "principal_axes": [
            [as_float(axis[0], 4), as_float(axis[1], 4), as_float(axis[2], 4), as_float(axis[3], 3)]
            for axis in reconstruction.get("principal_axes", [])
        ],
    }

    if vertices and faces and len(vertices) <= 12000 and len(faces) <= 24000:
        return ReconstructionViewerPayload(
            mode="mesh",
            vertices=[[as_float(axis, 4) for axis in vertex] for vertex in vertices],
            faces=[[int(index) for index in face] for face in faces],
            shell_vertices=[[as_float(axis, 4) for axis in vertex] for vertex in shell_vertices] if shell_vertices and len(shell_vertices) <= 18000 else [],
            shell_faces=[[int(index) for index in face] for face in shell_faces] if shell_faces and len(shell_faces) <= 36000 else [],
            **payload_kwargs,
        )

    points = np.argwhere(mask > 0)
    if len(points) > 2500:
        step = max(1, len(points) // 2500)
        points = points[::step]

    return ReconstructionViewerPayload(
        mode="point-cloud",
        points=[
            {
                "x": as_float(point[0] * spacing[0], 3),
                "y": as_float(point[1] * spacing[1], 3),
                "z": as_float(point[2] * spacing[2], 3),
            }
            for point in points
        ],
        **payload_kwargs,
    )


def build_pipeline(inference_mode: str) -> List[PipelineStep]:
    return [
        PipelineStep(step="Tiếp nhận và kiểm tra dữ liệu", status="completed"),
        PipelineStep(step="Chuẩn hóa thể tích ảnh", status="completed"),
        PipelineStep(step=f"Phân đoạn khối u ({inference_mode})", status="completed"),
        PipelineStep(step="Phân tích hình thái và đặc trưng ảnh", status="completed"),
        PipelineStep(step="Chấm điểm nguy cơ và xu hướng tiến triển", status="completed"),
        PipelineStep(step="Đóng gói báo cáo", status="completed"),
    ]


def confidence_band_from_scores(prediction_confidence: float, measurement_confidence: float) -> str:
    score = min(prediction_confidence, measurement_confidence)
    if score >= 90:
        return "rất cao"
    if score >= 80:
        return "cao"
    if score >= 68:
        return "trung bình"
    return "cần rà soát"


def segmentation_diagnostic_notes(
    *,
    mask_voxels: int,
    quality: float,
    uncertainty_score: float,
    connected_components: int,
    removed_components: int,
    prediction_confidence: float,
    measurement_confidence: float,
) -> List[str]:
    notes: List[str] = []
    if mask_voxels == 0:
        notes.append("Không phát hiện được vùng tổn thương rõ ràng trên phân đoạn hiện tại.")
    if prediction_confidence < 75:
        notes.append("Mức tin cậy dự đoán của mô hình chưa cao, nên xác nhận lại bằng mắt trên ảnh gốc.")
    if uncertainty_score >= 0.08:
        notes.append("Độ bất định của mô hình ở mức cao, nên kiểm tra thủ công trên lát cắt gốc.")
    if connected_components > 2:
        notes.append("Phân đoạn xuất hiện nhiều thành phần rời, cần xem lại tính liên tục giải phẫu.")
    if removed_components > 0:
        notes.append("Đã loại bỏ các thành phần nhỏ sau hậu xử lý để giảm nhiễu.")
    if quality < 0.72:
        notes.append("Chất lượng ca chưa lý tưởng cho việc diễn giải hoàn toàn tự động.")
    if measurement_confidence < 70:
        notes.append("Độ tin cậy đo đạc thấp, nên ưu tiên đối chiếu với chuyên gia chẩn đoán hình ảnh.")
    if not notes:
        notes.append("Phân đoạn ổn định, có thể dùng cho đánh giá hình thái và trực quan 3D.")
    return notes


def summarize_morphology(
    morphology_features: Dict[str, Any],
    reconstruction: Dict[str, Any],
) -> MorphologySummary:
    irregularity = max(
        float(reconstruction.get("surface_irregularity", 0.0)),
        float(morphology_features.get("surface_irregularity_index", 0.0)),
    )
    gradient_spread = float(morphology_features.get("gradient_distribution", 0.0))
    heterogeneity = float(morphology_features.get("intensity_heterogeneity_index", 0.0))
    fractal_dimension = float(
        morphology_features.get("fractal_dimension", reconstruction.get("fractal_dimension", 2.0))
    )
    contrast = float(morphology_features.get("glcm_contrast", 0.0))
    entropy = float(morphology_features.get("intensity_entropy", 0.0))
    asymmetry_score = float(morphology_features.get("asphericity", 0.0))

    if contrast > 55 or entropy > 4.8:
        texture_signature = "nhân không đồng nhất"
    elif contrast > 28 or gradient_spread > 35:
        texture_signature = "tín hiệu hỗn hợp"
    else:
        texture_signature = "bờ tương đối gọn"

    if asymmetry_score > 0.5:
        asymmetry = "cao"
    elif asymmetry_score > 0.28:
        asymmetry = "trung bình"
    else:
        asymmetry = "thấp"

    radial_heterogeneity = clamp((gradient_spread / 55) + (contrast / 180), 0.0, 1.0)
    margin_complexity = clamp((irregularity * 0.6) + ((fractal_dimension - 1.5) / 2.5), 0.0, 1.0)

    return MorphologySummary(
        surface_irregularity_index=as_float(irregularity, 4),
        fractal_dimension=as_float(fractal_dimension, 4),
        gradient_distribution=as_float(gradient_spread, 4),
        texture_signature=texture_signature,
        asymmetry=asymmetry,
        margin_complexity_index=as_float(margin_complexity, 4),
        radial_heterogeneity_index=as_float(radial_heterogeneity, 4),
        intensity_heterogeneity_index=as_float(heterogeneity, 4),
        volume_voxels=as_float(morphology_features.get("volume_voxels", 0.0), 2),
        surface_area_voxels=as_float(morphology_features.get("surface_area_voxels", 0.0), 2),
        elongation=as_float(morphology_features.get("elongation", 1.0), 4),
        flatness=as_float(morphology_features.get("flatness", 1.0), 4),
        compactness=as_float(morphology_features.get("compactness", 0.0), 4),
        sphericity=as_float(morphology_features.get("sphericity", reconstruction.get("sphericity", 0.0)), 4),
    )


def summarize_classification(
    classification_output: Dict[str, Any],
    morphology: MorphologySummary,
    quality: float,
) -> ClassificationResult:
    raw_label = classification_output.get("label", "uncertain")
    label = nhan_phan_loai(raw_label)

    risk_score = float(classification_output.get("risk_score", 0.0))
    confidence = float(classification_output.get("confidence", 0.5))
    reliability = clamp((quality * 0.58) + (confidence * 0.42), 0.5, 0.99)

    flags = []
    if risk_score >= 70:
        flags.append("Dấu hiệu ác tính ở mức cao")
    if morphology.surface_irregularity_index >= 0.35:
        flags.append("Bờ khối u không đều")
    if morphology.texture_signature != "bờ tương đối gọn":
        flags.append("Có hiện tượng không đồng nhất mô")
    if morphology.fractal_dimension >= 2.2:
        flags.append("Hình học đường bờ phức tạp")

    decision_factors = [item.strip() for item in classification_output.get("reasoning", "").split("|") if item.strip()]
    if not decision_factors:
        decision_factors = ["Đánh giá hình thái chuẩn"]

    return ClassificationResult(
        label=label,
        confidence=as_float(confidence, 4),
        risk_score=as_float(risk_score, 1),
        reliability_score=as_float(reliability, 4),
        reasoning=classification_output.get("reasoning", "Đánh giá hình thái chuẩn"),
        clinical_flags=flags[:4],
        decision_factors=decision_factors[:4],
    )


def collect_patient_history(patient_code: str, current_case_id: Optional[str] = None) -> List[Dict[str, Any]]:
    history = []
    for case in CASE_STORE.values():
        if case.get("patient", {}).get("code") != patient_code:
            continue
        if current_case_id and case.get("case_id") == current_case_id:
            continue
        history.append(case)
    history.sort(key=lambda item: item.get("created_at", ""))
    return history


def build_progression_from_history(
    patient_code: str,
    current_case_id: str,
    current_volume_cm3: float,
    current_morphology_features: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    analyzer = ProgressionAnalyzer()
    for historical_case in collect_patient_history(patient_code, current_case_id=current_case_id):
        historical_volume = float(historical_case.get("reconstruction", {}).get("volume_cm3", 0.0))
        if historical_volume <= 0:
            continue
        analyzer.add_timepoint(
            current_case_id,
            historical_volume,
            dict(historical_case.get("morphology", {})),
        )

    if current_volume_cm3 <= 0:
        return analyzer.predict_progression(current_case_id, 3), analyzer.predict_progression(current_case_id, 6)

    analyzer.add_timepoint(
        current_case_id,
        float(current_volume_cm3),
        dict(current_morphology_features),
    )
    return analyzer.predict_progression(current_case_id, 3), analyzer.predict_progression(current_case_id, 6)


def summarize_progression(
    prediction_3m: Dict[str, Any],
    prediction_6m: Dict[str, Any],
) -> ProgressionPrediction:
    velocity = float(prediction_6m.get("growth_velocity", 0.0))
    if velocity > 4.0:
        band = "tăng tốc"
    elif velocity > 1.0:
        band = "đang tăng"
    elif velocity < -1.0:
        band = "đang giảm"
    else:
        band = "ổn định"

    return ProgressionPrediction(
        volume_change_3m_percent=as_float(prediction_3m.get("volume_change_percent", 0.0), 2),
        volume_change_6m_percent=as_float(prediction_6m.get("volume_change_percent", 0.0), 2),
        growth_velocity=as_float(velocity, 3),
        growth_acceleration=as_float(prediction_6m.get("growth_acceleration", 0.0), 3),
        malignant_transition_risk=as_float(prediction_6m.get("malignant_risk", 0.0), 4),
        growth_velocity_band=band,
        forecast_summary=prediction_6m.get("summary", "Chưa đủ dữ liệu"),
    )


def summarize_treatment(
    classification: ClassificationResult,
    progression: ProgressionPrediction,
) -> TreatmentSimulation:
    risk_score = classification.risk_score
    growth_6m = progression.volume_change_6m_percent

    if risk_score >= 75 or growth_6m >= 25:
        strategy = "Hội chẩn u bướu khẩn"
        rationale = "Đặc trưng hình thái nguy cơ cao kèm xu hướng tăng thể tích ngắn hạn."
        confidence = 0.88
    elif risk_score >= 45 or growth_6m >= 10:
        strategy = "Theo dõi MRI ngắn hạn"
        rationale = "Các dấu hiệu nguy cơ trung bình cần theo dõi hình ảnh sát hơn."
        confidence = 0.78
    else:
        strategy = "Theo dõi định kỳ"
        rationale = "Hồ sơ hình ảnh hiện tại phù hợp với lịch theo dõi tiêu chuẩn."
        confidence = 0.71

    options = [
        TreatmentOption(
            name="Hội chẩn đa chuyên khoa",
            expected_volume_shift_percent=as_float(-min(18 + risk_score * 0.08, 30), 1),
            risk_reduction_score=as_float(min(35 + risk_score * 0.18, 70), 1),
        ),
        TreatmentOption(
            name="MRI ngắn hạn",
            expected_volume_shift_percent=as_float(-min(8 + max(growth_6m, 0) * 0.22, 18), 1),
            risk_reduction_score=as_float(min(18 + risk_score * 0.12, 42), 1),
        ),
        TreatmentOption(
            name="Theo dõi tiêu chuẩn",
            expected_volume_shift_percent=as_float(-min(4 + max(growth_6m, 0) * 0.1, 10), 1),
            risk_reduction_score=as_float(min(8 + risk_score * 0.06, 22), 1),
        ),
    ]

    return TreatmentSimulation(
        recommended_strategy=strategy,
        simulation_confidence=as_float(confidence, 4),
        rationale=rationale,
        options=options,
    )


def summarize_follow_up(
    classification: ClassificationResult,
    progression: ProgressionPrediction,
) -> Tuple[FollowUpPlan, TeleconsultationStatus]:
    today = date.today()
    if classification.risk_score >= 75 or progression.volume_change_6m_percent >= 25:
        offset = 7
        channel = "Hội chẩn u bướu + tư vấn từ xa"
        reason = "Cần ưu tiên xem xét vì nguy cơ cao và đang có dấu hiệu tăng trưởng hoạt động."
        tele_status = "đã lên lịch"
        note = "Hồ sơ đã sẵn sàng cho hội chẩn từ xa."
    elif classification.risk_score >= 45 or progression.volume_change_6m_percent >= 10:
        offset = 21
        channel = "Tái khám chẩn đoán hình ảnh"
        reason = "Mức nguy cơ trung bình cần rút ngắn khoảng cách theo dõi."
        tele_status = "sẵn sàng"
        note = "Có thể mở tư vấn từ xa nếu triệu chứng tiến triển."
    else:
        offset = 45
        channel = "Tái khám ngoại trú định kỳ"
        reason = "Xu hướng hiện tại đủ ổn định để theo dõi thường quy."
        tele_status = "tùy chọn"
        note = "Đánh giá từ xa là tùy chọn."

    follow_up = FollowUpPlan(
        next_recommended_date=(today + timedelta(days=offset)).isoformat(),
        channel=channel,
        reason=reason,
    )
    teleconsultation = TeleconsultationStatus(status=tele_status, note=note)
    return follow_up, teleconsultation


def build_follow_up_reminder(follow_up: FollowUpPlan) -> FollowUpReminder:
    target_date = datetime.fromisoformat(follow_up.next_recommended_date).date()
    today = date.today()
    delta_days = (target_date - today).days

    if delta_days < 0:
        severity = "quá hạn"
        message = f"Ca này đã quá lịch tái khám {-delta_days} ngày."
    elif delta_days <= 7:
        severity = "cần chuẩn bị"
        message = f"Cần chuẩn bị tái khám trong {delta_days} ngày tới."
    elif delta_days <= 21:
        severity = "sắp đến hạn"
        message = f"Lịch tái khám còn {delta_days} ngày."
    else:
        severity = "theo dõi"
        message = f"Chưa đến hạn, còn {delta_days} ngày tới mốc tái khám."

    return FollowUpReminder(
        due_in_days=int(delta_days),
        overdue=delta_days < 0,
        severity=severity,
        message=message,
    )


def build_patient_timeline(patient_code: str, current_case: Dict[str, Any]) -> List[TimelineSnapshot]:
    current_case_id = current_case.get("case_id")
    timeline_cases = collect_patient_history(patient_code, current_case_id=current_case_id)
    timeline_cases.append(current_case)
    timeline_cases = sorted(timeline_cases, key=lambda item: item.get("created_at", ""))

    snapshots = []
    for item in timeline_cases:
        snapshots.append(
            TimelineSnapshot(
                case_id=item["case_id"],
                date=item.get("date", today_iso()),
                modality=item.get("modality", item.get("study", {}).get("modality", "--")),
                tumor_volume_cm3=float(item.get("reconstruction", {}).get("volume_cm3", item.get("tumor_volume", 0.0))),
                risk_score=float(item.get("classification", {}).get("risk_score", 0.0)),
                review_status=item.get("ai_summary", {}).get("review_status", item.get("status", "--")),
            )
        )
    return snapshots


def build_teleconsultation_board(
    classification: ClassificationResult,
    progression: ProgressionPrediction,
    treatment: TreatmentSimulation,
    ai_summary: AISummary,
    segmentation: SegmentationResult,
    existing_notes: Optional[List[Dict[str, Any]]] = None,
) -> TeleconsultationBoard:
    specialists = ["Chẩn đoán hình ảnh", "Ngoại thần kinh"]
    if classification.risk_score >= 60:
        specialists.append("Ung bướu")
    if progression.volume_change_6m_percent >= 10:
        specialists.append("Xạ trị")

    if ai_summary.triage_level == "Nguy kịch":
        board_priority = "khẩn"
    elif ai_summary.triage_level == "Cao":
        board_priority = "ưu tiên"
    else:
        board_priority = "thường quy"

    summary = (
        f"Khối u {classification.label}, thể tích {segmentation.tumor_volume_cm3:.2f} cm3, "
        f"nguy cơ {classification.risk_score:.1f}/100, thay đổi 6 tháng {progression.volume_change_6m_percent:.1f}%, "
        f"hướng xử trí: {treatment.recommended_strategy.lower()}."
    )

    notes = [TeleconsultationNote(**note) for note in (existing_notes or [])]
    if not notes:
        notes.append(
            TeleconsultationNote(
                note_id=f"note_{uuid.uuid4().hex[:8]}",
                author="TumorSight AI",
                role="Tóm tắt tự động",
                content=summary,
                created_at=now_iso(),
                sign_off=False,
            )
        )

    return TeleconsultationBoard(
        packet_ready=bool(segmentation.tumor_volume_cm3 > 0 and ai_summary.review_status),
        board_priority=board_priority,
        recommended_specialists=specialists,
        summary=summary,
        notes=notes,
    )


def append_teleconsultation_note(
    case: Dict[str, Any],
    *,
    author: str,
    role: str,
    content: str,
    sign_off: bool,
) -> Dict[str, Any]:
    board = dict(case.get("teleconsultation_board", {}))
    notes = list(board.get("notes", []))
    notes.append(
        TeleconsultationNote(
            note_id=f"note_{uuid.uuid4().hex[:8]}",
            author=author.strip(),
            role=role.strip(),
            content=content.strip(),
            created_at=now_iso(),
            sign_off=bool(sign_off),
        ).model_dump(mode="json")
    )
    board["notes"] = notes
    case["teleconsultation_board"] = TeleconsultationBoard(**board).model_dump(mode="json")
    case["updated_at"] = now_iso()
    return case


def summarize_ai(
    classification: ClassificationResult,
    progression: ProgressionPrediction,
    treatment: TreatmentSimulation,
) -> Tuple[AISummary, InsightSummary]:
    risk_score = classification.risk_score
    growth_score = max(progression.volume_change_6m_percent, 0.0)

    if risk_score >= 80 or growth_score >= 30:
        review_status = "Sẵn sàng hội chẩn"
        triage_level = "Nguy kịch"
        portfolio_tag = "Ưu tiên hội chẩn"
        priority_band = "khẩn"
    elif risk_score >= 55 or growth_score >= 12:
        review_status = "Sẵn sàng rà soát"
        triage_level = "Cao"
        portfolio_tag = "Theo dõi sát"
        priority_band = "ưu tiên"
    else:
        review_status = "Sẵn sàng theo dõi"
        triage_level = "Trung bình"
        portfolio_tag = "Theo dõi định kỳ"
        priority_band = "thường quy"

    alerts = []
    if classification.risk_score >= 70:
        alerts.append("Điểm ác tính vượt ngưỡng nguy cơ cao nội bộ")
    if progression.malignant_transition_risk >= 0.45:
        alerts.append("Xu hướng theo thời gian cho thấy nguy cơ chuyển dạng đáng kể")
    if progression.growth_velocity_band == "tăng tốc":
        alerts.append("Tốc độ phát triển đang tăng nhanh")

    actions = [
        f"Rà soát gói phân đoạn để chuẩn bị cho hướng xử trí: {treatment.recommended_strategy.lower()}",
        "Kiểm tra chất lượng ảnh trước khi chốt báo cáo",
        "Xác nhận mốc tái khám với nhóm điều trị",
    ]
    if classification.label == "không xác định":
        actions.insert(0, "Đề nghị bác sĩ chẩn đoán hình ảnh rà soát thêm vì hình thái còn mơ hồ")

    consensus_score = clamp(
        (classification.reliability_score * 0.55)
        + ((1 - progression.malignant_transition_risk) * 0.15)
        + (0.30 if classification.label != "không xác định" else 0.18),
        0.45,
        0.98,
    )

    ai_summary = AISummary(
        review_status=review_status,
        triage_level=triage_level,
        consensus_score=as_float(consensus_score, 4),
        recommended_actions=actions[:4],
        alerts=alerts[:4],
    )
    insight = InsightSummary(portfolio_tag=portfolio_tag, priority_band=priority_band)
    return ai_summary, insight


def _normalize_preview_intensity(slice_array: np.ndarray) -> np.ndarray:
    image = np.asarray(slice_array, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.uint8)
    sample = image[finite]
    lower, upper = np.percentile(sample, [1.0, 99.0])
    if upper - lower < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    image = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _slice_for_plane(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unsupported plane: {plane}")


def _build_slice_preview(
    plane: str,
    volume: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    index: int,
) -> SegmentationSlicePreview:
    image_slice = _slice_for_plane(volume, plane, index)
    mask_slice = _slice_for_plane(mask, plane, index).astype(np.uint8)
    probability_slice = _slice_for_plane(probability, plane, index).astype(np.float32)
    normalized_image = _normalize_preview_intensity(image_slice)
    normalized_probability = np.clip(probability_slice * 255.0, 0, 255).astype(np.uint8)

    return SegmentationSlicePreview(
        plane=plane,
        width=int(mask_slice.shape[1]),
        height=int(mask_slice.shape[0]),
        slice_index=int(index),
        lesion_area_ratio=as_float(mask_slice.mean(), 4),
        image_pixels=normalized_image.ravel().astype(int).tolist(),
        mask_pixels=mask_slice.ravel().astype(int).tolist(),
        probability_pixels=normalized_probability.ravel().astype(int).tolist(),
    )


def _overlay_size_mm(plane: str, shape: Sequence[int], spacing: Sequence[float]) -> List[float]:
    depth, height, width = [int(item) for item in shape]
    spacing = [float(item) for item in spacing]
    if plane == "axial":
        return [max((width - 1) * spacing[2], spacing[2]), max((height - 1) * spacing[1], spacing[1])]
    if plane == "coronal":
        return [max((width - 1) * spacing[2], spacing[2]), max((depth - 1) * spacing[0], spacing[0])]
    if plane == "sagittal":
        return [max((depth - 1) * spacing[0], spacing[0]), max((height - 1) * spacing[1], spacing[1])]
    raise ValueError(f"Unsupported plane: {plane}")


def _overlay_position_mm(plane: str, shape: Sequence[int], spacing: Sequence[float], index: int) -> List[float]:
    depth, height, width = [int(item) for item in shape]
    spacing = [float(item) for item in spacing]
    center_x = ((width - 1) * spacing[2]) * 0.5
    center_y = ((height - 1) * spacing[1]) * 0.5
    center_z = ((depth - 1) * spacing[0]) * 0.5

    if plane == "axial":
        return [center_x, center_y, index * spacing[0]]
    if plane == "coronal":
        return [center_x, index * spacing[1], center_z]
    if plane == "sagittal":
        return [index * spacing[2], center_y, center_z]
    raise ValueError(f"Unsupported plane: {plane}")


def _best_overlay_indices(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    coords = np.argwhere(mask > 0)
    if len(coords):
        center = np.round(coords.mean(axis=0)).astype(int)
    else:
        center = np.asarray(mask.shape) // 2

    axial_scores = mask.sum(axis=(1, 2))
    coronal_scores = mask.sum(axis=(0, 2))
    sagittal_scores = mask.sum(axis=(0, 1))
    indices = {
        "axial": int(axial_scores.argmax()) if axial_scores.size and int(axial_scores.max()) > 0 else int(center[0]),
        "coronal": int(coronal_scores.argmax()) if coronal_scores.size and int(coronal_scores.max()) > 0 else int(center[1]),
        "sagittal": int(sagittal_scores.argmax()) if sagittal_scores.size and int(sagittal_scores.max()) > 0 else int(center[2]),
    }
    return center, indices


def build_viewer_overlay_payload(
    volume: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    voxel_spacing: Sequence[float],
) -> ViewerOverlayPayload:
    center, indices = _best_overlay_indices(mask)

    def overlay_for(plane: str) -> ViewerSliceOverlay:
        preview = _build_slice_preview(plane, volume, mask, probability, indices[plane])
        return ViewerSliceOverlay(
            plane=plane,
            width=preview.width,
            height=preview.height,
            slice_index=preview.slice_index,
            lesion_area_ratio=preview.lesion_area_ratio,
            position_mm=[as_float(value, 3) for value in _overlay_position_mm(plane, volume.shape, voxel_spacing, indices[plane])],
            size_mm=[as_float(value, 3) for value in _overlay_size_mm(plane, volume.shape, voxel_spacing)],
            image_pixels=preview.image_pixels,
            mask_pixels=preview.mask_pixels,
            probability_pixels=preview.probability_pixels,
        )

    return ViewerOverlayPayload(
        lesion_center_index=[int(item) for item in center.tolist()],
        axial=overlay_for("axial"),
        coronal=overlay_for("coronal"),
        sagittal=overlay_for("sagittal"),
    )


def _encode_uint8_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array.astype(np.uint8))
    return base64.b64encode(payload.tobytes()).decode("ascii")


def _normalize_volume_cube(volume: np.ndarray) -> np.ndarray:
    image = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.uint8)
    sample = image[finite]
    lower, upper = np.percentile(sample, [1.0, 99.0])
    if upper - lower < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def build_mpr_payload(
    volume: np.ndarray,
    mask: np.ndarray,
    probability: np.ndarray,
    voxel_spacing: Sequence[float],
    max_edge: int = 128,
) -> Dict[str, Any]:
    image = np.asarray(volume, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.uint8)
    probability_array = np.asarray(probability, dtype=np.float32)
    source_shape = np.asarray(image.shape, dtype=np.int32)
    max_dim = int(source_shape.max()) if source_shape.size else 0
    scale = min(1.0, float(max_edge) / max(max_dim, 1))

    if scale < 0.999:
        zoom = tuple(float(scale) for _ in range(3))
        image = ndimage.zoom(image, zoom=zoom, order=1)
        mask_array = ndimage.zoom(mask_array.astype(np.float32), zoom=zoom, order=0).astype(np.uint8)
        probability_array = ndimage.zoom(probability_array, zoom=zoom, order=1).astype(np.float32)
        display_spacing = [as_float(float(spacing) / scale, 5) for spacing in voxel_spacing]
    else:
        display_spacing = [as_float(float(spacing), 5) for spacing in voxel_spacing]

    center, overlay_indices = _best_overlay_indices(mask_array)
    bbox_coords = np.argwhere(mask_array > 0)
    if len(bbox_coords):
        bbox_min = bbox_coords.min(axis=0).astype(int).tolist()
        bbox_max = bbox_coords.max(axis=0).astype(int).tolist()
    else:
        bbox_min = [0, 0, 0]
        bbox_max = (np.asarray(mask_array.shape, dtype=np.int32) - 1).tolist()

    return {
        "shape": [int(item) for item in image.shape],
        "source_shape": [int(item) for item in source_shape.tolist()],
        "voxel_spacing_mm": display_spacing,
        "default_index": [int(item) for item in center.tolist()],
        "recommended_index": [
            int(overlay_indices["axial"]),
            int(overlay_indices["coronal"]),
            int(overlay_indices["sagittal"]),
        ],
        "lesion_bbox_min": bbox_min,
        "lesion_bbox_max": bbox_max,
        "volume_base64": _encode_uint8_array(_normalize_volume_cube(image)),
        "mask_base64": _encode_uint8_array(np.clip(mask_array, 0, 1) * 255),
        "probability_base64": _encode_uint8_array(np.clip(probability_array, 0.0, 1.0) * 255.0),
    }


def build_preview(volume: np.ndarray, mask: np.ndarray, probability: np.ndarray) -> SegmentationPreview:
    coords = np.argwhere(mask > 0)
    if len(coords):
        center = np.round(coords.mean(axis=0)).astype(int)
        bbox_min = coords.min(axis=0).astype(int)
        bbox_max = coords.max(axis=0).astype(int)
    else:
        center = np.asarray(mask.shape) // 2
        bbox_min = np.zeros(3, dtype=int)
        bbox_max = np.asarray(mask.shape, dtype=int) - 1

    _, overlay_indices = _best_overlay_indices(mask)
    axial_scores = mask.sum(axis=(1, 2))
    coronal_scores = mask.sum(axis=(0, 2))
    sagittal_scores = mask.sum(axis=(0, 1))
    plane_scores = {
        "axial": int(axial_scores.max()) if axial_scores.size else 0,
        "coronal": int(coronal_scores.max()) if coronal_scores.size else 0,
        "sagittal": int(sagittal_scores.max()) if sagittal_scores.size else 0,
    }
    dominant_plane = max(plane_scores, key=plane_scores.get)

    return SegmentationPreview(
        dominant_plane=dominant_plane,
        lesion_center_index=[int(item) for item in center.tolist()],
        lesion_bbox_min=[int(item) for item in bbox_min.tolist()],
        lesion_bbox_max=[int(item) for item in bbox_max.tolist()],
        axial=_build_slice_preview("axial", volume, mask, probability, overlay_indices["axial"]),
        coronal=_build_slice_preview("coronal", volume, mask, probability, overlay_indices["coronal"]),
        sagittal=_build_slice_preview("sagittal", volume, mask, probability, overlay_indices["sagittal"]),
    )


def build_segmentation_artifacts(
    case: Dict[str, Any],
    mask: np.ndarray,
    probability: np.ndarray,
    preview: SegmentationPreview,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    positive = int(mask.sum())
    probability_positive = probability[mask > 0]
    prob_mean = float(probability_positive.mean()) if probability_positive.size else 0.0
    coords = np.argwhere(mask > 0)
    sample_points = coords[:: max(1, len(coords) // 250)].tolist() if len(coords) else []

    mask_artifact = {
        "case_id": case["case_id"],
        "shape": list(mask.shape),
        "voxel_spacing_mm": case["preprocessing"]["resample_spacing_mm"],
        "tumor_voxel_count": positive,
        "positive_ratio": as_float(positive / mask.size, 6),
        "lesion_extent_mm": case["segmentation"]["lesion_extent_mm"],
        "sample_points": sample_points[:250],
        "preview": preview.model_dump(mode="json"),
    }
    probability_artifact = {
        "case_id": case["case_id"],
        "shape": list(probability.shape),
        "voxel_spacing_mm": case["preprocessing"]["resample_spacing_mm"],
        "probability_min": as_float(probability.min(), 6),
        "probability_max": as_float(probability.max(), 6),
        "probability_mean": as_float(probability.mean(), 6),
        "positive_region_mean": as_float(prob_mean, 6),
        "preview": preview.model_dump(mode="json"),
    }
    return mask_artifact, probability_artifact


def generate_report(case: Dict[str, Any]) -> str:
    segmentation = case["segmentation"]
    classification = case["classification"]
    progression = case["progression"]
    morphology = case["morphology"]
    return "\n".join(
        [
            f"# Báo cáo TumorSight: {case['patient_name']}",
            "",
            f"- Mã ca: `{case['case_id']}`",
            f"- Ngày: {case['date']}",
            f"- Phương thức chụp: {case['modality']}",
            f"- Trạng thái rà soát: {case['ai_summary']['review_status']}",
            "",
            "## Phân đoạn",
            f"- Mô hình: {segmentation['model_name']}",
            f"- Thể tích: {segmentation['tumor_volume_cm3']:.2f} cm3",
            f"- Độ ổn định Dice nội bộ: {segmentation['dice_score']:.4f}",
            f"- Độ ổn định IoU nội bộ: {segmentation['iou']:.4f}",
            f"- Mức tin cậy: {segmentation.get('confidence_band', '--')}",
            f"- Kích thước tổn thương (mm): {' x '.join(str(item) for item in segmentation['lesion_extent_mm'])}",
            f"- Ghi chú chất lượng: {'; '.join(segmentation.get('diagnostic_notes', [])[:2]) or '--'}",
            "",
            "## Hình thái",
            f"- Độ không đều bề mặt: {morphology['surface_irregularity_index']:.4f}",
            f"- Chiều fractal: {morphology['fractal_dimension']:.4f}",
            f"- Dấu hiệu cấu trúc: {morphology['texture_signature']}",
            "",
            "## Nguy cơ",
            f"- Phân loại: {classification['label']}",
            f"- Điểm nguy cơ: {classification['risk_score']:.1f}",
            f"- Độ tin cậy: {classification['confidence']:.4f}",
            "",
            "## Tiến triển",
            f"- Thay đổi sau 3 tháng: {progression['volume_change_3m_percent']:.2f}%",
            f"- Thay đổi sau 6 tháng: {progression['volume_change_6m_percent']:.2f}%",
            f"- Nguy cơ chuyển dạng: {progression['malignant_transition_risk']:.4f}",
            f"- Dự báo: {progression['forecast_summary']}",
            "",
            "## Kế hoạch",
            f"- Hướng xử trí khuyến nghị: {case['treatment_simulation']['recommended_strategy']}",
            f"- Tái khám: {case['follow_up']['next_recommended_date']} qua {case['follow_up']['channel']}",
            f"- Nhắc lịch: {case.get('follow_up_reminder', {}).get('message', '--')}",
            f"- Ưu tiên hội chẩn: {case.get('teleconsultation_board', {}).get('board_priority', '--')}",
        ]
    )


def persist_case(case: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    case_id = case["case_id"]
    CASE_STORE[case_id] = case
    np.savez_compressed(
        arrays_path(case_id),
        volume=arrays["volume"],
        source_volume=arrays.get("source_volume", arrays["volume"]),
        mask=arrays["mask"],
        probability=arrays["probability"],
        spacing=arrays.get("spacing", np.asarray(VOXEL_SPACING, dtype=np.float32)),
    )
    ARRAY_CACHE[case_id] = arrays

    preview = build_preview(arrays["volume"], arrays["mask"], arrays["probability"])
    viewer_overlays = build_viewer_overlay_payload(
        arrays["volume"],
        arrays["mask"],
        arrays["probability"],
        arrays.get("spacing", np.asarray(VOXEL_SPACING, dtype=np.float32)),
    )
    mpr_payload = build_mpr_payload(
        arrays["volume"],
        arrays["mask"],
        arrays["probability"],
        arrays.get("spacing", np.asarray(VOXEL_SPACING, dtype=np.float32)),
    )
    mask_payload, probability_payload = build_segmentation_artifacts(case, arrays["mask"], arrays["probability"], preview)
    write_json(bundle_path(case_id), case)
    write_json(preview_path(case_id), preview.model_dump(mode="json"))
    write_json(viewer_overlay_artifact_path(case_id), viewer_overlays.model_dump(mode="json"))
    write_json(mpr_artifact_path(case_id), mpr_payload)
    write_json(mask_artifact_path(case_id), mask_payload)
    write_json(probability_artifact_path(case_id), probability_payload)
    report_path(case_id).write_text(generate_report(case), encoding="utf-8")


def load_saved_cases() -> None:
    CASE_STORE.clear()
    for path in CASE_ARTIFACTS_DIR.glob("*/bundle.json"):
        try:
            array_file = path.parent / "arrays.npz"
            if not array_file.exists():
                continue
            archive = np.load(array_file)
            if "source_volume" not in archive.files:
                continue
            case = read_json(path)
            if "case_id" in case:
                CASE_STORE[case["case_id"]] = case
        except Exception as exc:
            logger.warning("Skipping corrupt case bundle %s: %s", path, exc)


def create_case_record(
    patient_name: str,
    modality: str,
    timepoint_count: int,
    file_count: int,
    source_files: Sequence[str],
    *,
    case_id: Optional[str] = None,
    patient_code: Optional[str] = None,
    hospital: Optional[str] = None,
    input_volume: Optional[np.ndarray] = None,
    voxel_spacing: Optional[Sequence[float]] = None,
    source_type: Optional[str] = None,
    series_labels: Optional[Sequence[str]] = None,
    ingestion_extras: Optional[Dict[str, Any]] = None,
    analysis_mode: str = "full",
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if input_volume is None:
        raise ValueError("Khong co volume thuc de phan tich")

    created_at = now_iso()
    case_id = case_id or f"case_{uuid.uuid4().hex[:8]}"
    patient_code = patient_code or patient_code_from_name(patient_name)
    file_names = list(source_files) or ["uploaded-study.dcm"]
    spacing = tuple(float(item) for item in (voxel_spacing or VOXEL_SPACING))
    volume = np.asarray(input_volume, dtype=np.float32)
    prediction = run_segmentation_with_best_model(
        volume,
        voxel_spacing=spacing,
        series_labels=series_labels,
        runtime_mode=analysis_mode,
    )
    if prediction.mask.ndim != 3 or prediction.probability.ndim != 3:
        raise ValueError("Mo hinh phan doan khong tra ve mask 3D hop le")

    mask = prediction.mask.astype(np.uint8)
    probability = prediction.probability.astype(np.float32)
    threshold = float(prediction.threshold)
    quality = float(prediction.quality_score)

    if volume.ndim == 4:
        display_volume = np.mean(volume[: min(volume.shape[0], 4)], axis=0).astype(np.float32)
    else:
        display_volume = volume.astype(np.float32)

    reconstructor = Reconstruction3D(voxel_spacing=spacing)
    morphology_analyzer = AdvancedMorphologyAnalyzer(voxel_spacing=spacing)
    reconstruction_analysis = reconstructor.analyze(mask, probability=probability, threshold=threshold)
    viewer_payload = build_viewer_payload(reconstruction_analysis, mask)

    morphology_features = morphology_analyzer.analyze(display_volume, mask)
    morphology_features["surface_irregularity"] = max(
        float(morphology_features.get("surface_irregularity", 0.0)),
        float(reconstruction_analysis.get("surface_irregularity", 0.0)),
    )
    morphology_features["volume_cm3"] = float(reconstruction_analysis.get("volume_cm3", 0.0))

    classification_output = classifier_engine().classify(dict(morphology_features), volume=volume, mask=mask)
    prediction_3m, prediction_6m = build_progression_from_history(
        patient_code,
        case_id,
        float(reconstruction_analysis.get("volume_cm3", 0.0)),
        morphology_features,
    )

    inferred_formats = infer_format_labels(file_names)
    inference_mode = prediction.inference_mode
    extent_mm = lesion_extent_mm(mask, spacing)
    normalized_modalities = [normalize_brats_modality(label) for label in (series_labels or ())]
    available_modalities = sorted({label for label in normalized_modalities if label})
    if not available_modalities and volume.ndim == 4 and volume.shape[0] >= 4:
        available_modalities = list(BRATS_MODALITIES)
    missing_modalities = [label for label in BRATS_MODALITIES if label not in available_modalities]
    modality_completeness = len(available_modalities) / len(BRATS_MODALITIES) if available_modalities else 0.0
    internal_dice = float(prediction.metadata.get("internal_dice_score", quality))
    internal_iou = float(prediction.metadata.get("internal_iou_score", (internal_dice / (2 - internal_dice)) if internal_dice < 2 else 0.0))
    hausdorff = float(prediction.metadata.get("cleanup_hausdorff_voxels", 0.0)) * float(np.mean(spacing))
    stability_score = float(prediction.metadata.get("stability_score", quality))
    uncertainty_score = float(prediction.metadata.get("mean_uncertainty", 0.0))
    connected_components = int(prediction.metadata.get("connected_components", 0))
    removed_components = int(prediction.metadata.get("removed_components", 0))
    prediction_confidence = clamp(float(prediction.confidence_score), 0.0, 99.0)
    measurement_confidence = clamp((stability_score * 100.0) - min(hausdorff * 3.5, 16.0), 55.0, 99.0)
    data_quality_score = clamp((quality * 100.0) - (uncertainty_score * 1000.0), 0.0, 99.0)
    if modality_completeness < 1.0:
        penalty = (1.0 - modality_completeness)
        prediction_confidence = clamp(prediction_confidence - (penalty * 18.0), 0.0, 99.0)
        measurement_confidence = clamp(measurement_confidence - (penalty * 14.0), 0.0, 99.0)
        data_quality_score = clamp(data_quality_score - (penalty * 22.0), 0.0, 99.0)
    confidence_band = confidence_band_from_scores(prediction_confidence, measurement_confidence)
    diagnostic_notes = segmentation_diagnostic_notes(
        mask_voxels=int(mask.sum()),
        quality=quality,
        uncertainty_score=uncertainty_score,
        connected_components=connected_components,
        removed_components=removed_components,
        prediction_confidence=prediction_confidence,
        measurement_confidence=measurement_confidence,
    )
    if analysis_mode == "fast":
        diagnostic_notes.append("Che do fast mode: giam so TTA view va metric nang de tang toc do xu ly.")
    if missing_modalities:
        diagnostic_notes.append(
            "Thiếu chuỗi MRI chuẩn cho BraTS: " + ", ".join(label.upper() for label in missing_modalities) + "."
        )
    if prediction.model_key != DA_NNUNET_MODEL_KEY and modality_completeness < 1.0:
        diagnostic_notes.append("Đã dùng mô hình fallback do study chưa đủ 4 modality chuẩn cho DA-nnUNet.")
    source_kind = source_type or ("uploaded_dicom" if "DICOM" in inferred_formats else "uploaded_volume")

    patient = PatientInfo(
        name=patient_name,
        code=patient_code,
        age=None,
        sex=None,
        hospital=hospital or "Chưa cung cấp",
    )
    study = StudyInfo(
        modality=modality,
        uploaded_at=created_at,
        region="Não",
        timepoints=timepoint_count,
        study_uid=f"TS-{case_id}",
    )
    ingestion = IngestionInfo(
        source_type=source_kind,
        file_count=file_count,
        readiness_score=as_float(quality * 100, 1),
        accepted_formats=inferred_formats,
        series_count=len(series_labels or []),
        series_labels=[str(item) for item in (series_labels or [])],
        total_upload_size_mb=float((ingestion_extras or {}).get("total_upload_size_mb", 0.0)),
        deidentification_applied=bool((ingestion_extras or {}).get("deidentification_applied", False)),
        deidentified_tags=[str(item) for item in (ingestion_extras or {}).get("deidentified_tags", [])],
        upload_warnings=[str(item) for item in (ingestion_extras or {}).get("upload_warnings", [])],
    )
    if analysis_mode == "fast":
        ingestion.upload_warnings.append("Case duoc xu ly bang fast mode de toi uu thoi gian.")
    preprocessing = PreprocessingInfo(
        resample_spacing_mm=[float(item) for item in spacing],
        intensity_normalization="chuẩn hóa theo từng kênh bằng percentile + z-score",
        roi_strategy="giữ nguyên không gian ảnh gốc, không nội suy giả lập",
        data_quality_score=as_float(data_quality_score, 1),
    )
    review_required = confidence_band == "cần rà soát" or bool(missing_modalities)
    segmentation = SegmentationResult(
        model=prediction.model_key,
        model_name=prediction.model_name,
        dice_score=as_float(internal_dice, 4),
        iou=as_float(internal_iou, 4),
        threshold=as_float(threshold, 4),
        case_quality_score=as_float(quality, 4),
        tumor_volume_cm3=as_float(reconstruction_analysis.get("volume_cm3", 0.0), 3),
        hausdorff_mm=as_float(hausdorff, 3),
        prediction_confidence_score=as_float(prediction_confidence, 1),
        measurement_confidence_score=as_float(measurement_confidence, 1),
        lesion_extent_mm=extent_mm,
        inference_shape=list(mask.shape),
        inference_mode=inference_mode,
        confidence_band=confidence_band,
        review_recommended=review_required,
        uncertainty_score=as_float(uncertainty_score, 6),
        diagnostic_notes=diagnostic_notes,
        mask_path=f"{API_PREFIX}/cases/{case_id}/segmentation/mask",
        probability_map_path=f"{API_PREFIX}/cases/{case_id}/segmentation/probability-map",
    )
    morphology = summarize_morphology(morphology_features, reconstruction_analysis)
    classification = summarize_classification(classification_output, morphology, quality)
    progression = summarize_progression(prediction_3m, prediction_6m)
    treatment = summarize_treatment(classification, progression)
    follow_up, teleconsultation = summarize_follow_up(classification, progression)
    ai_summary, insight = summarize_ai(classification, progression, treatment)
    follow_up_reminder = build_follow_up_reminder(follow_up)
    teleconsultation_board = build_teleconsultation_board(
        classification,
        progression,
        treatment,
        ai_summary,
        segmentation,
    )
    reconstruction = ReconstructionResult(
        volume_cm3=as_float(reconstruction_analysis.get("volume_cm3", 0.0), 3),
        surface_area_cm2=as_float(reconstruction_analysis.get("surface_area_cm2", 0.0), 3),
        sphericity=as_float(reconstruction_analysis.get("sphericity", 0.0), 4),
        surface_irregularity=as_float(reconstruction_analysis.get("surface_irregularity", 0.0), 4),
        fractal_dimension=as_float(reconstruction_analysis.get("fractal_dimension", 0.0), 4),
        viewer_payload=viewer_payload,
    )
    case_context = {
        "case_id": case_id,
        "created_at": created_at,
        "date": today_iso(),
        "modality": modality,
        "tumor_volume": segmentation.tumor_volume_cm3,
        "patient": {"code": patient_code},
        "reconstruction": {"volume_cm3": reconstruction.volume_cm3},
        "classification": {"risk_score": classification.risk_score},
        "ai_summary": {"review_status": ai_summary.review_status},
    }
    patient_timeline = build_patient_timeline(patient_code, case_context)

    case = CaseRecord(
        id=case_id,
        case_id=case_id,
        created_at=created_at,
        updated_at=created_at,
        date=today_iso(),
        status=ai_summary.review_status,
        patient_name=patient_name,
        modality=modality,
        tumor_volume=segmentation.tumor_volume_cm3,
        tumor_surface=reconstruction.surface_area_cm2,
        tumor_density=as_float(float(probability[mask > 0].mean()) if int(mask.sum()) else 0.0, 4),
        patient=patient,
        study=study,
        ingestion=ingestion,
        preprocessing=preprocessing,
        pipeline=build_pipeline(inference_mode),
        segmentation=segmentation,
        morphology=morphology,
        classification=classification,
        progression=progression,
        reconstruction=reconstruction,
        treatment_simulation=treatment,
        follow_up=follow_up,
        teleconsultation=teleconsultation,
        teleconsultation_board=teleconsultation_board,
        follow_up_reminder=follow_up_reminder,
        patient_timeline=patient_timeline,
        ai_summary=ai_summary,
        insight=insight,
    ).model_dump(mode="json")

    arrays = {
        "volume": display_volume,
        "source_volume": volume.astype(np.float32),
        "mask": mask.astype(np.uint8),
        "probability": probability.astype(np.float32),
        "spacing": np.asarray(spacing, dtype=np.float32),
    }
    return case, arrays


@app.on_event("startup")
async def startup_event() -> None:
    load_saved_cases()
    load_runtime_models(force=True)
    for model in runtime_model_registry():
        logger.info(
            "Model runtime %s: %s | ready=%s | mode=%s",
            model["task"],
            model["name"],
            model["ready"],
            model["runtime_mode"],
        )
    logger.info("TumorSight da san sang voi %s ca", len(CASE_STORE))


@app.get("/")
async def root(request: Request):
    if request.query_params:
        logger.info("Lam sach URL goc co query string cu: %s", str(request.url))
        return RedirectResponse(url="/", status_code=307, headers=NO_CACHE_HEADERS)
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers=NO_CACHE_HEADERS)
    return {"message": "Nen tang TumorSight da san sang"}


@app.get(f"{API_PREFIX}/platform")
async def get_platform_info() -> Dict[str, Any]:
    return build_platform_payload()


@app.get(f"{API_PREFIX}/platform/health")
async def get_health_status() -> Dict[str, Any]:
    return build_platform_payload()["health"]


@app.get(f"{API_PREFIX}/segmentation/models")
async def list_segmentation_models() -> Dict[str, Any]:
    load_runtime_models()
    models = []
    preferred = Path(BEST_CHECKPOINT_PATH)
    for key in available_model_keys():
        best_file = get_model_artifact_path(key)
        models.append(
            {
                "key": key,
                "name": model_name_from_key(key),
                "status": "ready" if best_file.exists() else "missing",
                "status_label": "sẵn sàng" if best_file.exists() else "thiếu mô hình",
                "preferred": best_file == preferred,
            }
        )
    return {"models": models}


@app.get(f"{API_PREFIX}/models")
async def list_runtime_models() -> Dict[str, Any]:
    return {"models": runtime_model_registry()}


@app.get(f"{API_PREFIX}/cases")
async def list_cases() -> List[Dict[str, Any]]:
    return sorted(CASE_STORE.values(), key=lambda item: item.get("updated_at", ""), reverse=True)


@app.get(f"{API_PREFIX}/cases/{{case_id}}")
async def get_case(case_id: str) -> Dict[str, Any]:
    return get_case_or_404(case_id)


@app.get(f"{API_PREFIX}/cases/{{case_id}}/timeline")
async def get_case_timeline(case_id: str) -> List[Dict[str, Any]]:
    case = get_case_or_404(case_id)
    patient_code = case.get("patient", {}).get("code")
    if not patient_code:
        return []
    current_case = {
        "case_id": case["case_id"],
        "created_at": case.get("created_at", now_iso()),
        "date": case.get("date", today_iso()),
        "modality": case.get("modality", case.get("study", {}).get("modality", "--")),
        "tumor_volume": case.get("tumor_volume", 0.0),
        "reconstruction": {"volume_cm3": case.get("reconstruction", {}).get("volume_cm3", 0.0)},
        "classification": {"risk_score": case.get("classification", {}).get("risk_score", 0.0)},
        "ai_summary": {"review_status": case.get("ai_summary", {}).get("review_status", case.get("status", "--"))},
    }
    timeline = build_patient_timeline(patient_code, current_case)
    case["patient_timeline"] = [item.model_dump(mode="json") for item in timeline]
    return case["patient_timeline"]


@app.get(f"{API_PREFIX}/cases/{{case_id}}/teleconsultation/board")
async def get_teleconsultation_board(case_id: str) -> Dict[str, Any]:
    return get_case_or_404(case_id).get("teleconsultation_board", {})


@app.post(f"{API_PREFIX}/cases/{{case_id}}/teleconsultation/notes")
async def add_teleconsultation_note(
    case_id: str,
    author: str = Form(...),
    role: str = Form(...),
    content: str = Form(...),
    sign_off: bool = Form(False),
) -> Dict[str, Any]:
    if not author.strip() or not role.strip() or not content.strip():
        raise HTTPException(status_code=400, detail="Tac gia, vai tro va noi dung ghi chu khong duoc de trong")
    case = get_case_or_404(case_id)
    updated_case = append_teleconsultation_note(
        case,
        author=author,
        role=role,
        content=content,
        sign_off=sign_off,
    )
    persist_case(updated_case, load_case_arrays(case_id))
    return updated_case["teleconsultation_board"]


@app.post(f"{API_PREFIX}/cases")
async def create_case(
    patient_name: str = Form("Bệnh nhân"),
    modality: str = Form("MRI"),
    timepoint_count: int = Form(1),
) -> Dict[str, Any]:
    raise HTTPException(
        status_code=400,
        detail="Route nay khong tao ca gia lap. Hay dung /api/analyze voi study thuc te.",
    )


@app.post(f"{API_PREFIX}/analyze")
async def analyze_case(
    patient_name: str = Form("Bệnh nhân mới"),
    modality: str = Form("MRI"),
    timepoint_count: int = Form(1),
    analysis_mode: str = Form("full"),
    dicom_files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    valid_files = [upload for upload in dicom_files if upload and upload.filename]
    if not valid_files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một tệp ảnh cho ca chụp")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    ANALYSIS_JOBS[job_id] = AnalysisJob(
        job_id=job_id,
        status="running",
        progress=25,
        message=f"Dang chay suy luan bang {BEST_CHECKPOINT_NAME}",
    ).model_dump(mode="json")

    try:
        upload_dir = TMP_UPLOAD_DIR / "uploads" / job_id
        saved_paths = await store_uploaded_files(valid_files, upload_dir)
        ingestion_extras = build_ingestion_safety_summary(saved_paths)
        uploaded_study = load_uploaded_study(saved_paths)
        if uploaded_study is None:
            raise HTTPException(status_code=400, detail="Khong doc duoc volume 3D hop le tu study da tai len")
        case, arrays = create_case_record(
            patient_name=patient_name,
            modality=modality,
            timepoint_count=max(1, timepoint_count),
            file_count=len(valid_files),
            source_files=[upload.filename or "study.dcm" for upload in valid_files],
            input_volume=uploaded_study.volume,
            voxel_spacing=uploaded_study.voxel_spacing_mm,
            source_type=uploaded_study.source_type,
            series_labels=uploaded_study.series_labels,
            ingestion_extras=ingestion_extras,
            analysis_mode="fast" if str(analysis_mode).lower() == "fast" else "full",
        )
        CASE_STORE[case["case_id"]] = case
        persist_case(case, arrays)
        ANALYSIS_JOBS[job_id] = AnalysisJob(
            job_id=job_id,
            status="completed",
            progress=100,
            message="Phan tich da hoan tat",
            case_id=case["case_id"],
        ).model_dump(mode="json")
        return {"job_id": job_id, "case_id": case["case_id"], "case": case}
    except HTTPException as exc:
        ANALYSIS_JOBS[job_id] = AnalysisJob(
            job_id=job_id,
            status="failed",
            progress=100,
            message=str(exc.detail),
        ).model_dump(mode="json")
        raise
    except Exception as exc:
        logger.exception("Phan tich that bai")
        ANALYSIS_JOBS[job_id] = AnalysisJob(
            job_id=job_id,
            status="failed",
            progress=100,
            message=str(exc),
        ).model_dump(mode="json")
        raise HTTPException(status_code=500, detail="Phan tich that bai") from exc


@app.post(f"{API_PREFIX}/cases/{{case_id}}/segment")
async def run_segmentation(
    case_id: str,
    analysis_mode: str = Form("full"),
    dicom_files: Optional[List[UploadFile]] = File(None),
) -> Dict[str, Any]:
    current_case = get_case_or_404(case_id)
    valid_files = [upload for upload in (dicom_files or []) if upload and upload.filename]
    file_names = [upload.filename or "study.dcm" for upload in valid_files] or []
    file_count = len(file_names) if valid_files else int(current_case["ingestion"]["file_count"])
    uploaded_study = None

    if valid_files:
        upload_dir = TMP_UPLOAD_DIR / "uploads" / case_id
        saved_paths = await store_uploaded_files(valid_files, upload_dir)
        ingestion_extras = build_ingestion_safety_summary(saved_paths)
        uploaded_study = load_uploaded_study(saved_paths)
        if uploaded_study is None:
            raise HTTPException(status_code=400, detail="Khong doc duoc volume 3D hop le tu study da tai len")
    else:
        stored_arrays = load_case_arrays(case_id)
        if "source_volume" not in stored_arrays:
            raise HTTPException(status_code=400, detail="Khong con source volume de chay lai phan doan")
        inferred_source_type = current_case.get("ingestion", {}).get("source_type", "uploaded_volume")
        uploaded_study = StudyVolume(
            volume=np.asarray(stored_arrays["source_volume"], dtype=np.float32),
            voxel_spacing_mm=tuple(float(item) for item in stored_arrays.get("spacing", np.asarray(VOXEL_SPACING))),
            source_type=inferred_source_type,
            series_labels=tuple(current_case.get("ingestion", {}).get("series_labels", [])),
        )
        file_names = ["stored-study.dcm"] if inferred_source_type == "uploaded_dicom" else ["stored-study.nii.gz"]
        ingestion_extras = {
            "total_upload_size_mb": float(current_case.get("ingestion", {}).get("total_upload_size_mb", 0.0)),
            "deidentification_applied": bool(current_case.get("ingestion", {}).get("deidentification_applied", False)),
            "deidentified_tags": list(current_case.get("ingestion", {}).get("deidentified_tags", [])),
            "upload_warnings": list(current_case.get("ingestion", {}).get("upload_warnings", [])),
        }

    refreshed_case, arrays = create_case_record(
        patient_name=current_case["patient_name"],
        modality=current_case["modality"],
        timepoint_count=int(current_case["study"]["timepoints"]),
        file_count=file_count,
        source_files=file_names,
        case_id=case_id,
        patient_code=current_case["patient"]["code"],
        hospital=current_case["patient"]["hospital"],
        input_volume=uploaded_study.volume,
        voxel_spacing=uploaded_study.voxel_spacing_mm,
        source_type=uploaded_study.source_type,
        series_labels=uploaded_study.series_labels or current_case.get("ingestion", {}).get("series_labels", []),
        ingestion_extras=ingestion_extras,
        analysis_mode="fast" if str(analysis_mode).lower() == "fast" else "full",
    )
    CASE_STORE[case_id] = refreshed_case
    persist_case(refreshed_case, arrays)
    return {"status": "success", "status_label": "thành công", "segmentation": refreshed_case["segmentation"], "case": refreshed_case}


@app.get(f"{API_PREFIX}/cases/{{case_id}}/morphology")
async def get_morphology_analysis(case_id: str) -> Dict[str, Any]:
    return get_case_or_404(case_id)["morphology"]


@app.get(f"{API_PREFIX}/cases/{{case_id}}/classification")
async def get_classification(case_id: str) -> Dict[str, Any]:
    return get_case_or_404(case_id)["classification"]


@app.get(f"{API_PREFIX}/cases/{{case_id}}/progression")
async def get_progression_prediction(case_id: str) -> Dict[str, Any]:
    return get_case_or_404(case_id)["progression"]


@app.get(f"{API_PREFIX}/cases/{{case_id}}/3d-data")
async def get_3d_visualization_data(case_id: str) -> Dict[str, Any]:
    case = get_case_or_404(case_id)
    reconstruction = case["reconstruction"]
    return {
        "case_id": case_id,
        "tumor_volume": case["tumor_volume"],
        "tumor_surface": case["tumor_surface"],
        "sphericity": reconstruction["sphericity"],
        "confidence": case["segmentation"]["dice_score"],
        **reconstruction["viewer_payload"],
    }


@app.get(f"{API_PREFIX}/cases/{{case_id}}/segmentation/preview")
async def get_segmentation_preview(case_id: str) -> Dict[str, Any]:
    get_case_or_404(case_id)
    artifact = preview_path(case_id)
    if artifact.exists():
        cached = read_json(artifact)
        if all(key in cached for key in ("axial", "coronal", "sagittal")):
            return cached
    arrays = load_case_arrays(case_id)
    preview = build_preview(arrays["volume"], arrays["mask"], arrays["probability"]).model_dump(mode="json")
    write_json(artifact, preview)
    return preview


@app.get(f"{API_PREFIX}/cases/{{case_id}}/mpr")
async def get_case_mpr(case_id: str) -> Dict[str, Any]:
    get_case_or_404(case_id)
    artifact = mpr_artifact_path(case_id)
    if artifact.exists():
        return read_json(artifact)
    arrays = load_case_arrays(case_id)
    payload = build_mpr_payload(
        arrays["volume"],
        arrays["mask"],
        arrays["probability"],
        arrays.get("spacing", np.asarray(VOXEL_SPACING, dtype=np.float32)),
    )
    write_json(artifact, payload)
    return payload


@app.get(f"{API_PREFIX}/cases/{{case_id}}/viewer/orthogonal-overlays")
async def get_viewer_overlays(case_id: str) -> Dict[str, Any]:
    get_case_or_404(case_id)
    artifact = viewer_overlay_artifact_path(case_id)
    if artifact.exists():
        return read_json(artifact)
    arrays = load_case_arrays(case_id)
    payload = build_viewer_overlay_payload(
        arrays["volume"],
        arrays["mask"],
        arrays["probability"],
        arrays.get("spacing", np.asarray(VOXEL_SPACING, dtype=np.float32)),
    ).model_dump(mode="json")
    write_json(artifact, payload)
    return payload


@app.get(f"{API_PREFIX}/cases/{{case_id}}/segmentation/mask")
async def download_mask_artifact(case_id: str):
    get_case_or_404(case_id)
    artifact = mask_artifact_path(case_id)
    if not artifact.exists():
        arrays = load_case_arrays(case_id)
        preview = build_preview(arrays["volume"], arrays["mask"], arrays["probability"])
        payload, _ = build_segmentation_artifacts(get_case_or_404(case_id), arrays["mask"], arrays["probability"], preview)
        write_json(artifact, payload)
    return FileResponse(artifact, media_type="application/json", filename=f"{case_id}_mask.json")


@app.get(f"{API_PREFIX}/cases/{{case_id}}/segmentation/probability-map")
async def download_probability_artifact(case_id: str):
    get_case_or_404(case_id)
    artifact = probability_artifact_path(case_id)
    if not artifact.exists():
        arrays = load_case_arrays(case_id)
        preview = build_preview(arrays["volume"], arrays["mask"], arrays["probability"])
        _, payload = build_segmentation_artifacts(get_case_or_404(case_id), arrays["mask"], arrays["probability"], preview)
        write_json(artifact, payload)
    return FileResponse(artifact, media_type="application/json", filename=f"{case_id}_probability_map.json")


@app.get(f"{API_PREFIX}/cases/{{case_id}}/bundle.json")
async def download_case_bundle(case_id: str):
    get_case_or_404(case_id)
    artifact = bundle_path(case_id)
    if not artifact.exists():
        write_json(artifact, get_case_or_404(case_id))
    return FileResponse(artifact, media_type="application/json", filename=f"{case_id}_bundle.json")


@app.get(f"{API_PREFIX}/cases/{{case_id}}/report.md")
async def download_case_report(case_id: str):
    case = get_case_or_404(case_id)
    artifact = report_path(case_id)
    if not artifact.exists():
        artifact.write_text(generate_report(case), encoding="utf-8")
    return PlainTextResponse(artifact.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    job = ANALYSIS_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Khong tim thay tien trinh")
    return job


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=DEBUG)
