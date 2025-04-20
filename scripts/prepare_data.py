"""
数据准备脚本 - 下载并处理蒸馏所需的数据
"""

import os
import argparse
import logging
import sys

# 添加项目根目录到路径
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
    parser = argparse.ArgumentParser(description="准备知识蒸馏所需的数据")
    parser.add_argument("--num_samples", type=int, default=1000,
                      help="处理的样本数量，默认为1000，设为-1表示处理全部")
    parser.add_argument("--output_dir", type=str, default=config.PROCESSED_DATA_DIR,
                      help=f"处理后的数据保存目录，默认为{config.PROCESSED_DATA_DIR}")
    parser.add_argument("--cache_dir", type=str, default=config.CACHE_DIR,
                      help=f"缓存目录，默认为{config.CACHE_DIR}")
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 下载数据集
    logger.info("开始下载数据集")
    dataset = download_dataset()
    
    # 下载模型
    logger.info("开始下载模型")
    teacher_path, _ = download_models()
    
    # 使用教师模型生成带思考过程的输出
    logger.info("使用教师模型生成思考过程")
    num_samples = None if args.num_samples == -1 else args.num_samples
    teacher_outputs = generate_teacher_outputs(
        teacher_model_path=teacher_path,
        dataset=dataset,
        num_samples=num_samples
    )
    
    # 准备蒸馏数据集
    logger.info("准备蒸馏数据集")
    prepare_distillation_dataset(
        teacher_outputs=teacher_outputs,
        output_dir=args.output_dir
    )
    
    logger.info(f"数据准备完成，已保存到 {args.output_dir}")

if __name__ == "__main__":
    main()