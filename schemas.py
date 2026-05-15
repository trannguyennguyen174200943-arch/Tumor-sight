"""Lược đồ Pydantic cho API TumorSight - Định nghĩa cấu trúc dữ liệu"""

from typing import Dict, List, Optional  # Các type hint

from pydantic import BaseModel, Field  # Pydantic cho validation dữ liệu


class PatientInfo(BaseModel):
    """Thông tin bệnh nhân"""
    name: str  # Tên bệnh nhân
    code: Optional[str] = None  # Mã bệnh nhân
    age: Optional[int] = None  # Tuổi
    sex: Optional[str] = None  # Giới tính
    hospital: Optional[str] = None  # Bệnh viện


class StudyInfo(BaseModel):
    """Thông tin bộ hình ảnh y tế"""
    modality: str  # Phương thức chụp (MRI, CT, v.v.)
    uploaded_at: str  # Thời gian upload
    region: str = "Brain"  # Vùng cơ thể (mặc định: não)
    timepoints: int = 1  # Số điểm thời gian
    study_uid: Optional[str] = None  # ID duy nhất của bộ hình ảnh


class IngestionInfo(BaseModel):
    """Thông tin nhập liệu file"""
    source_type: str  # Loại nguồn (DICOM, NIfTI, v.v.)
    file_count: int  # Số file
    readiness_score: float  # Điểm sẵn sàng (0-1)
    accepted_formats: List[str] = Field(default_factory=list)  # Các định dạng chấp nhận
    series_count: int = 0  # Số chuỗi ảnh
    series_labels: List[str] = Field(default_factory=list)  # Nhãn các chuỗi (T1, T2, v.v.)
    total_upload_size_mb: float = 0.0  # Tổng dung lượng upload (MB)
    deidentification_applied: bool = False  # Có áp dụng ẩn danh metadata hay không
    deidentified_tags: List[str] = Field(default_factory=list)  # Danh sách tag đã ẩn danh
    upload_warnings: List[str] = Field(default_factory=list)  # Cảnh báo phát sinh trong lúc ingest


class PreprocessingInfo(BaseModel):
    """Thông tin tiền xử lý ảnh"""
    resample_spacing_mm: List[float] = Field(default_factory=list)  # Kích thước voxel sau resample
    intensity_normalization: str  # Phương pháp chuẩn hóa cường độ
    roi_strategy: str  # Chiến lược xác định vùng quan tâm
    data_quality_score: float  # Điểm chất lượng dữ liệu


class PipelineStep(BaseModel):
    """Bước trong quy trình xử lý"""
    step: str  # Tên bước
    status: str  # Trạng thái (pending, running, completed, failed)


class SegmentationResult(BaseModel):
    """Kết quả phân đoạn u não"""
    model: str  # Khóa model
    model_name: str  # Tên model
    dice_score: float  # Điểm DICE (độ chính xác phân đoạn)
    iou: float  # IoU score (Intersection over Union)
    threshold: float  # Ngưỡng quyết định
    case_quality_score: float  # Điểm chất lượng trường hợp
    tumor_volume_cm3: float  # Thể tích u (cm³)
    hausdorff_mm: float  # Khoảng cách Hausdorff
    prediction_confidence_score: float  # Điểm tin cậy dự đoán
    measurement_confidence_score: float  # Điểm tin cậy đo lường
    lesion_extent_mm: List[float] = Field(default_factory=list)  # Kích thước u (mm)
    inference_shape: List[int] = Field(default_factory=list)  # Kích thước input mô hình
    inference_mode: str  # Chế độ suy diễn
    confidence_band: Optional[str] = None  # Dải tin cậy
    review_recommended: bool = False  # Có cần xem xét lại không
    uncertainty_score: Optional[float] = None  # Điểm không chắc chắn
    diagnostic_notes: List[str] = Field(default_factory=list)  # Ghi chú chẩn đoán
    mask_path: Optional[str] = None  # Đường dẫn file mặt nạ
    probability_map_path: Optional[str] = None  # Đường dẫn bản đồ xác suất


class MorphologySummary(BaseModel):
    """Tóm tắt hình thái học của u"""
    surface_irregularity_index: float  # Chỉ số độ bất thường bề mặt
    fractal_dimension: float  # Chiều fractal (độ phức tạp)
    gradient_distribution: float  # Phân bố gradient
    texture_signature: str  # Chữ ký kết cấu
    asymmetry: str  # Tính không đối xứng
    margin_complexity_index: float  # Chỉ số độ phức tạp biên
    radial_heterogeneity_index: float  # Chỉ số không đồng nhất hướng tâm
    intensity_heterogeneity_index: float  # Chỉ số không đồng nhất cường độ
    volume_voxels: float  # Thể tích (voxel)
    surface_area_voxels: float  # Diện tích bề mặt (voxel)
    elongation: float  # Độ kéo dài
    flatness: float  # Độ phẳng
    compactness: float  # Độ nhỏ gọn
    sphericity: float  # Độ hình cầu


class ClassificationResult(BaseModel):
    """Kết quả phân loại u (lành tính/ác tính)"""
    label: str  # Nhãn (benign/malignant/uncertain)
    confidence: float  # Mức tin cậy (0-1)
    risk_score: float  # Điểm rủi ro
    reliability_score: float  # Điểm độ tin cậy
    reasoning: str  # Lý do phân loại
    clinical_flags: List[str] = Field(default_factory=list)  # Các dấu hiệu lâm sàng
    decision_factors: List[str] = Field(default_factory=list)  # Các yếu tố quyết định


class ProgressionPrediction(BaseModel):
    """Dự đoán tiến triển u"""
    volume_change_3m_percent: float  # Thay đổi thể tích sau 3 tháng (%)
    volume_change_6m_percent: float  # Thay đổi thể tích sau 6 tháng (%)
    growth_velocity: float  # Tốc độ tăng trưởng
    growth_acceleration: float  # Gia tốc tăng trưởng
    malignant_transition_risk: float  # Nguy cơ chuyển thành ác tính
    growth_velocity_band: str  # Dải tốc độ tăng trưởng
    forecast_summary: str  # Tóm tắt dự báo


class ReconstructionViewerPayload(BaseModel):
    """Dữ liệu tái dựng 3D cho viewer"""
    mode: str  # Chế độ hiển thị (mesh, voxel, v.v.)
    vertices: List[List[float]] = Field(default_factory=list)  # Danh sách đỉnh mesh
    faces: List[List[int]] = Field(default_factory=list)  # Danh sách mặt mesh
    shell_vertices: List[List[float]] = Field(default_factory=list)  # Đỉnh bao ngoài
    shell_faces: List[List[int]] = Field(default_factory=list)  # Mặt bao ngoài
    points: List[Dict[str, float]] = Field(default_factory=list)  # Các điểm đặc biệt
    bounds_min_mm: List[float] = Field(default_factory=list)  # Tọa độ góc nhỏ nhất
    bounds_max_mm: List[float] = Field(default_factory=list)  # Tọa độ góc lớn nhất
    centroid_mm: List[float] = Field(default_factory=list)  # Tâm khối lượng
    dimensions_mm: List[float] = Field(default_factory=list)  # Kích thước
    voxel_spacing_mm: List[float] = Field(default_factory=list)  # Kích thước voxel
    principal_axes: List[List[float]] = Field(default_factory=list)  # Trục chính
    mesh_quality_score: float = 0.0  # Điểm chất lượng mesh
    vertex_count: int = 0  # Số đỉnh
    face_count: int = 0  # Số mặt


class ViewerSliceOverlay(BaseModel):
    """Overlay lát cắt 2D trên viewer"""
    plane: str  # Mặt phẳng (axial, coronal, sagittal)
    width: int  # Chiều rộng ảnh (pixel)
    height: int  # Chiều cao ảnh (pixel)
    slice_index: int  # Chỉ số lát cắt
    lesion_area_ratio: float  # Tỷ lệ diện tích u
    position_mm: List[float] = Field(default_factory=list)  # Vị trí (mm)
    size_mm: List[float] = Field(default_factory=list)  # Kích thước (mm)
    image_pixels: List[int] = Field(default_factory=list)  # Dữ liệu ảnh (RGB pixels)
    mask_pixels: List[int] = Field(default_factory=list)  # Dữ liệu mặt nạ
    probability_pixels: List[int] = Field(default_factory=list)  # Dữ liệu xác suất


class ViewerOverlayPayload(BaseModel):
    """Dữ liệu overlay 3D cho viewer"""
    lesion_center_index: List[int] = Field(default_factory=list)  # Vị trí trung tâm u
    axial: ViewerSliceOverlay  # Overlay view trục
    coronal: ViewerSliceOverlay  # Overlay view trực đứng
    sagittal: ViewerSliceOverlay  # Overlay view bên


class ReconstructionResult(BaseModel):
    """Kết quả tái dựng 3D"""
    volume_cm3: float  # Thể tích (cm³)
    surface_area_cm2: float  # Diện tích bề mặt (cm²)
    sphericity: float  # Độ hình cầu
    surface_irregularity: float  # Độ bất thường bề mặt
    fractal_dimension: float  # Chiều fractal
    viewer_payload: ReconstructionViewerPayload  # Dữ liệu viewer


class TreatmentOption(BaseModel):
    name: str
    expected_volume_shift_percent: float
    risk_reduction_score: float


class TreatmentSimulation(BaseModel):
    recommended_strategy: str
    simulation_confidence: float
    rationale: str
    options: List[TreatmentOption] = Field(default_factory=list)


class FollowUpPlan(BaseModel):
    next_recommended_date: str
    channel: str
    reason: str


class TeleconsultationStatus(BaseModel):
    status: str
    note: Optional[str] = None


class TeleconsultationNote(BaseModel):
    note_id: str
    author: str
    role: str
    content: str
    created_at: str
    sign_off: bool = False


class TeleconsultationBoard(BaseModel):
    packet_ready: bool
    board_priority: str
    recommended_specialists: List[str] = Field(default_factory=list)
    summary: str
    notes: List[TeleconsultationNote] = Field(default_factory=list)


class TimelineSnapshot(BaseModel):
    case_id: str
    date: str
    modality: str
    tumor_volume_cm3: float
    risk_score: float
    review_status: str


class FollowUpReminder(BaseModel):
    due_in_days: int
    overdue: bool
    severity: str
    message: str


class AISummary(BaseModel):
    review_status: str
    triage_level: str
    consensus_score: float
    recommended_actions: List[str] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)


class InsightSummary(BaseModel):
    portfolio_tag: str
    priority_band: str


class SegmentationSlicePreview(BaseModel):
    plane: str
    width: int
    height: int
    slice_index: int
    lesion_area_ratio: float
    image_pixels: List[int] = Field(default_factory=list)
    mask_pixels: List[int] = Field(default_factory=list)
    probability_pixels: List[int] = Field(default_factory=list)


class SegmentationPreview(BaseModel):
    dominant_plane: str
    lesion_center_index: List[int] = Field(default_factory=list)
    lesion_bbox_min: List[int] = Field(default_factory=list)
    lesion_bbox_max: List[int] = Field(default_factory=list)
    axial: SegmentationSlicePreview
    coronal: SegmentationSlicePreview
    sagittal: SegmentationSlicePreview


class CaseRecord(BaseModel):
    id: str
    case_id: str
    created_at: str
    updated_at: str
    date: str
    status: str
    patient_name: str
    modality: str
    tumor_volume: float
    tumor_surface: float
    tumor_density: float
    patient: PatientInfo
    study: StudyInfo
    ingestion: IngestionInfo
    preprocessing: PreprocessingInfo
    pipeline: List[PipelineStep] = Field(default_factory=list)
    segmentation: SegmentationResult
    morphology: MorphologySummary
    classification: ClassificationResult
    progression: ProgressionPrediction
    reconstruction: ReconstructionResult
    treatment_simulation: TreatmentSimulation
    follow_up: FollowUpPlan
    teleconsultation: TeleconsultationStatus
    teleconsultation_board: TeleconsultationBoard
    follow_up_reminder: FollowUpReminder
    patient_timeline: List[TimelineSnapshot] = Field(default_factory=list)
    ai_summary: AISummary
    insight: InsightSummary


class AnalysisJob(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    case_id: Optional[str] = None


class PlatformHealthCheckpoints(BaseModel):
    preferred_model_key: str
    preferred_model_name: str
    preferred_model_path: str
    training_checkpoint_ready: bool
    training_checkpoint_size_mb: Optional[float] = None
    available_models: List[str] = Field(default_factory=list)


class PlatformHealthChecks(BaseModel):
    api: str
    storage: str
    dataset: str
    checkpoints: PlatformHealthCheckpoints


class PlatformHealth(BaseModel):
    status: str
    generated_at: str
    checks: PlatformHealthChecks


class TrainingEvidence(BaseModel):
    best_val_dice: Optional[float] = None
    epochs: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None


class EvaluationEvidence(BaseModel):
    clean_mae_cm3: Optional[float] = None
    clean_mape_percent: Optional[float] = None
    raw_mape_percent: Optional[float] = None
    validation_cases: Optional[int] = None


class PlatformEvidence(BaseModel):
    training: TrainingEvidence
    evaluation: EvaluationEvidence


class PortfolioSummary(BaseModel):
    case_count: int
    avg_completion_score: float
    avg_priority_score: float
    ready_for_board: int
    needs_attention: int
    top_case_ids: List[str] = Field(default_factory=list)


class PlatformInfo(BaseModel):
    name: str
    version: str
    availability: bool
    gpu: bool
    memory: str
    case_count: int
    health: PlatformHealth
    evidence: PlatformEvidence
    portfolio: PortfolioSummary
