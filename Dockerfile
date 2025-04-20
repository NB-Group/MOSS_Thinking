FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露API服务端口
EXPOSE 8000

# 设置启动命令
CMD ["python", "service.py", "--model_path", "/models/output", "--port", "8000"]