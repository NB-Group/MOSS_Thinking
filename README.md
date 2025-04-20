# Qwen "边想边说"项目生产部署指南

这个项目实现了一个基于Qwen模型的"边想边说"功能，使模型能够展示思考过程并支持中断交互。

## 环境配置

### 1. 使用Conda创建环境

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate moss_thinking
```

### 2. 手动安装依赖（如果不使用Conda）

```bash
# 创建虚拟环境
python -m venv moss_env
source moss_env/bin/activate  # Linux/Mac
moss_env\Scripts\activate  # Windows

# 安装依赖
pip install torch==2.1.0 numpy==1.24.3
pip install transformers>=4.35.0 peft>=0.5.0 accelerate>=0.23.0 bitsandbytes>=0.41.0
pip install datasets>=2.14.0 tqdm>=4.66.0 sentencepiece>=0.1.99 modelscope>=1.9.0
pip install huggingface-hub>=0.17.0 safetensors>=0.3.3 jsonlines>=3.1.0 colorama>=0.4.6
```

## 运行步骤

### 1. 数据准备

此步骤使用教师模型(DeepSeek-R1-Distill-Llama-70B-AWQ)生成带思考过程的数据：

```bash
python main.py prepare --num_samples 1000 --output_dir ./data/processed
```

参数说明：
- `--num_samples`：处理的样本数量，默认为1000，设为-1表示处理全部
- `--output_dir`：处理后的数据保存目录

### 2. 模型训练

训练微调Qwen模型，使其具备"边想边说"的能力：

```bash
python main.py train --data_dir ./data/processed --output_dir ./output
```

参数说明：
- `--data_dir`：数据目录，包含train.json和val.json
- `--output_dir`：模型保存目录
- `--model_path`：（可选）指定本地模型路径
- `--no_lora`：（可选）不使用LoRA进行微调

### 3. 生产环境部署

对于生产环境，使用 `service.py` 脚本启动API服务：

```bash
python service.py --model_path ./output --port 8000
```

参数说明：
- `--model_path`：模型路径，默认为./output
- `--port`：服务端口，默认为8000
- `--host`：服务主机地址，默认为0.0.0.0（对外提供服务）

### 4. 快速体验（非生产环境）

如果只是想快速体验效果，可以运行交互式推理脚本：

```bash
python main.py infer --local_model --output_dir ./output
```

参数说明：
- `--local_model`：使用本地保存的模型
- `--output_dir`：本地模型的目录

## 系统要求

- CUDA兼容的GPU（建议至少12GB显存）
- 至少16GB系统内存
- 约50GB的硬盘空间（用于缓存模型和数据）

## 生产环境注意事项

1. 考虑将模型量化为int8或int4以减少资源占用
2. 使用docker部署以确保环境一致性
3. 配置负载均衡以处理高并发请求
4. 监控系统资源使用情况，尤其是GPU内存
5. 对API端点进行访问控制和速率限制
