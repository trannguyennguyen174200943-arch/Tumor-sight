# TumorSight

Nền tảng FastAPI + frontend tĩnh để mô phỏng quy trình phân tích u não 3D: tiếp nhận hồ sơ, segmentation, morphology, phân loại nguy cơ, dự báo tiến triển, dựng hình 3D và xuất bundle báo cáo.

## Thành phần chính

- `app.py`: backend FastAPI đã khôi phục, bám đúng API mà giao diện hiện tại gọi.
- `schemas.py`: schema Pydantic cho platform, case bundle, preview và artifacts.
  - Định nghĩa cấu trúc dữ liệu cho tất cả API responses
  - Bao gồm: PatientInfo, StudyInfo, SegmentationResult, ClassificationResult, v.v.
- `modules/`: các module segmentation, reconstruction, morphology, classification, progression.
  - `segmentation.py`: Phân đoạn u từ MRI (nnU-Net, MONAI models)
  - `reconstruction.py`: Tái dựng 3D mesh từ mặt nạ phân đoạn
  - `morphology.py`: Trích xuất đặc trưng hình thái (shape, texture, radiomics)
  - `classification.py`: Phân loại u (lành tính vs ác tính)
  - `progression.py`: Dự đoán tiến triển u (3-6 tháng)
- `static/`: dashboard web, viewer 3D Three.js và các asset giao diện.
  - `index.html`: Giao diện chính
  - `viewer3d.js`: Viewer 3D sử dụng Three.js
  - `app.js`: Logic frontend chính
- `tmp/cases/`: nơi backend lưu bundle JSON, preview, mask/probability artifact và report Markdown cho từng ca.
- `config.py`: Cấu hình tập trung cho toàn dự án
  - Đường dẫn model, directories, cấu hình API, v.v.

## Luồng hiện tại

1. Người dùng upload study qua `POST /api/analyze`.
2. Backend tạo volume/mask/probability tổng hợp, sau đó chạy reconstruction, morphology, classification và progression.
3. Kết quả được đóng gói thành một case hoàn chỉnh, lưu vào `tmp/cases/<case_id>/`.
4. Frontend đọc lại qua các route `platform`, `cases`, `preview`, `bundle`, `report`.

## API chính

- `GET /api/platform` - Lấy thông tin nền tảng
- `GET /api/platform/health` - Kiểm tra sức khỏe hệ thống
- `GET /api/cases` - Danh sách tất cả trường hợp
- `GET /api/cases/{case_id}` - Chi tiết một trường hợp
- `POST /api/cases` - Tạo trường hợp mới
- `POST /api/analyze` - Tải lên và phân tích study
- `POST /api/cases/{case_id}/segment` - Phân đoạn u
- `GET /api/cases/{case_id}/morphology` - Trích xuất hình thái
- `GET /api/cases/{case_id}/classification` - Phân loại u
- `GET /api/cases/{case_id}/progression` - Dự đoán tiến triển
- `GET /api/cases/{case_id}/3d-data` - Dữ liệu 3D reconstruction
- `GET /api/cases/{case_id}/segmentation/preview` - Xem trước phân đoạn
- `GET /api/cases/{case_id}/segmentation/mask` - Tải mask
- `GET /api/cases/{case_id}/segmentation/probability-map` - Tải bản đồ xác suất
- `GET /api/cases/{case_id}/bundle.json` - Tải bundle dữ liệu
- `GET /api/cases/{case_id}/report.md` - Tải báo cáo markdown
- `GET /api/jobs/{job_id}` - Kiểm tra trạng thái công việc

## Chạy dự án

```bash
pip install -r requirement.txt  # Cài đặt dependencies
python main.py  # Khởi động server FastAPI
```

Mặc định server chạy tại `http://127.0.0.1:8000`.

## Benchmark segmentation

Để đo `Dice / IoU / Hausdorff` trên bộ case có label thật và sinh gợi ý tuning threshold/ensemble:

```bash
python evaluate/benchmark_segmentation.py --dataset-root data\brats_peds_prepared
```

Yêu cầu tên file theo kiểu BraTS quen thuộc, ví dụ:
- `BraTS-PED-00001-000-t1c.nii.gz` (T1 tương phản)
- `BraTS-PED-00001-000-t1.nii.gz` (T1 không tương phản)
- `BraTS-PED-00001-000-t2.nii.gz` (T2)
- `BraTS-PED-00001-000-flair.nii.gz` hoặc `...-t2f.nii.gz` (FLAIR)
- `BraTS-PED-00001-000-seg.nii.gz` (Ground truth segmentation)

Script sẽ tạo:
- `checkpoints/benchmark/segmentation_eval_summary.json` - Tóm tắt kết quả đánh giá
- `checkpoints/benchmark/segmentation_tuning.json` - Gợi ý tuning threshold/ensemble

Backend sẽ tự đọc `segmentation_eval_summary.json` để cập nhật phần evidence trên dashboard.
Nếu có `checkpoints/benchmark/segmentation_tuning.json`, runtime segmentation sẽ tự nạp threshold/ensemble weight đã tune.

## Ghi chú

- Khi ứng dụng khởi động, nếu chưa có case đã lưu, backend sẽ seed một số case demo.
- Checkpoint ưu tiên được đọc từ `SEGMENTATION_CHECKPOINT_PATH` hoặc tự chọn trong `checkpoints/`.
- Viewer 3D nhận mesh hoặc point cloud trực tiếp từ `reconstruction.viewer_payload`.
- Tất cả dữ liệu bệnh nhân được lưu cục bộ trong `tmp/cases/` - hệ thống này không kết nối cơ sở dữ liệu từ xa.
