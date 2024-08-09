# Chọn phiên bản Python mong muốn
FROM python:3.10

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp cấu hình phụ thuộc
COPY requirements.txt .

# Cài đặt các gói cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Chạy ứng dụng với gunicorn
CMD ["gunicorn", "main:app"]
