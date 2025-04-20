"""
处理数据，使用教师模型生成带思考过程的数据
"""

import os
import json
import argparse
import logging
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.data_processing import (
    download_dataset, 
    download_models, 
    generate_teacher_outputs, 
    prepare_distillation_dataset
)
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument("--num_samples", type=int, default=1000,
                      help="使用的样本数量，默认为1000，设为-1表示全部")
    parser.add_argument("--output_dir", type=str, default=config.PROCESSED_DATA_DIR,
                      help=f"处理后的数据保存目录，默认为{config.PROCESSED_DATA_DIR}")
    
    args = parser.parse_args()
    num_samples = None if args.num_samples == -1 else args.num_samples
    
    # 下载数据集
    logger.info("开始下载数据集")
    dataset = download_dataset()
    
    # 修改：不下载模型和生成思考过程，直接处理数据
    logger.info("使用已带有思考过程的数据集")
    
    # 划分和保存训练集、验证集
    logger.info("准备蒸馏数据集")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 直接处理下载的数据
    train_data = []
    val_data = []
    
    # 从下载的数据集中选择样本
    samples = dataset["train"]
    if num_samples is not None:
        samples = samples[:num_samples]
    
    # 划分训练集和验证集
    val_size = min(config.VAL_SET_SIZE, int(len(samples) * 0.1))
    
    for i, item in enumerate(samples):
        # 数据集中已有的item格式: {"input": "用户问题", "output": "<think>思考过程</think>回答"}
        if i < val_size:
            val_data.append(item)
        else:
            train_data.append(item)
    
    # 保存为JSON文件
    with open(os.path.join(args.output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存 {len(train_data)} 条训练数据和 {len(val_data)} 条验证数据到 {args.output_dir}")
    
    logger.info(f"数据准备完成，已保存到 {args.output_dir}")

if __name__ == "__main__":
    main()