# Chọn base image Python 3.11 (TF 2.19 hỗ trợ tốt)
FROM python:3.11-slim

# Thiết lập working directory
WORKDIR /app

# Copy requirements và cài đặt dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code & model
COPY . .

# Khai báo port mà Streamlit lắng nghe
EXPOSE 8501

# Lệnh khởi chạy app
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
