"""
TumorSight - Điểm khởi động chính của ứng dụng
Khởi động server FastAPI bằng uvicorn
"""
import uvicorn  # Framework web để chạy API server
from config import HOST, PORT, DEBUG  # Lấy cấu hình từ file config

if __name__ == "__main__":
    # Khởi động server
    uvicorn.run(
        "app:app",  # Chỉ định module và ứng dụng Flask (app:app)
        host=HOST,  # Host của server (mặc định: 127.0.0.1)
        port=PORT,  # Cổng của server (mặc định: 8000)
        reload=DEBUG,  # Tự động reload nếu code thay đổi (chỉ khi DEBUG=True)
        log_level="info"  # Mức độ chi tiết của log
    )
