FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY cloudbaserun/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目（含 shared/）
COPY . .

# 环境变量
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "cloudbaserun.app:app", "--host", "0.0.0.0", "--port", "8000"]
