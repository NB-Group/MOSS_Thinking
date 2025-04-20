"""
项目配置文件
"""

# 模型配置
TEACHER_MODEL_ID = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"  # 教师模型
STUDENT_MODEL_ID = "Qwen/Qwen1.5-14B-Chat"  # 学生模型
OUTPUT_DIR = "./output"  # 输出目录

# 数据集配置
DATASET_ID = "liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT"  # 数据集ID
PROCESSED_DATA_DIR = "./data/processed"  # 处理后的数据目录
CACHE_DIR = "./cache"  # 缓存目录

# 训练配置
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.1
MAX_SOURCE_LENGTH = 512  # 输入最大长度
MAX_TARGET_LENGTH = 1024  # 输出最大长度
VAL_SET_SIZE = 2000  # 验证集大小
LOGGING_STEPS = 10  # 日志记录频率

# LoRA配置
USE_LORA = True  # 是否使用LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 思考标记
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"

# 特殊标记
CODE_RUN_TOKEN = "'''RUN"