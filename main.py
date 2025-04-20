"""
主入口文件 - 提供统一的命令行界面执行整个流程
"""

import os
import argparse
import logging
import sys
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_command(command):
    """运行命令并返回结果"""
    logger.info(f"执行命令: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"命令执行失败: {stderr}")
        return False, stderr
    
    return True, stdout

def main():
    parser = argparse.ArgumentParser(description="Qwen边想边说 - 知识蒸馏训练流程")
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 准备数据
    prepare_parser = subparsers.add_parser("prepare", help="准备训练数据")
    prepare_parser.add_argument("--num_samples", type=int, default=1000,
                              help="处理的样本数量，默认为1000，设为-1表示处理全部")
    prepare_parser.add_argument("--output_dir", type=str, default="./data/processed",
                              help="处理后的数据保存目录")
    
    # 训练模型
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model_path", type=str, default=None,
                            help="学生模型的路径，默认使用配置文件中指定的模型")
    train_parser.add_argument("--data_dir", type=str, default="./data/processed",
                            help="处理后的数据目录")
    train_parser.add_argument("--output_dir", type=str, default="./output",
                            help="输出目录")
    train_parser.add_argument("--no_lora", action="store_true",
                            help="不使用LoRA进行微调，默认使用")
    
    # 推理演示
    infer_parser = subparsers.add_parser("infer", help="运行推理演示")
    infer_parser.add_argument("--model_path", type=str, default=None,
                            help="模型路径，如果不指定则使用配置中的模型")
    infer_parser.add_argument("--local_model", action="store_true",
                            help="使用本地保存的模型，默认为False")
    
    # 全流程
    all_parser = subparsers.add_parser("all", help="执行完整流程：准备数据->训练模型->推理演示")
    all_parser.add_argument("--num_samples", type=int, default=1000,
                          help="处理的样本数量，默认为1000，设为-1表示处理全部")
    all_parser.add_argument("--no_lora", action="store_true",
                          help="不使用LoRA进行微调，默认使用")
    
    args = parser.parse_args()
    
    # 处理命令
    if args.command == "prepare":
        cmd = f"python scripts/prepare_data.py --num_samples {args.num_samples} --output_dir {args.output_dir}"
        success, output = run_command(cmd)
        if not success:
            return
        
    elif args.command == "train":
        cmd = f"python scripts/train_model.py --data_dir {args.data_dir} --output_dir {args.output_dir}"
        if args.model_path:
            cmd += f" --model_path {args.model_path}"
        if args.no_lora:
            cmd += " --no_lora"
        success, output = run_command(cmd)
        if not success:
            return
        
    elif args.command == "infer":
        cmd = "python scripts/inference.py"
        if args.model_path:
            cmd += f" --model_path {args.model_path}"
        if args.local_model:
            cmd += " --local_model"
        success, output = run_command(cmd)
        if not success:
            return
        
    elif args.command == "all":
        # 1. 准备数据
        logger.info("步骤 1/3: 准备数据")
        cmd = f"python scripts/prepare_data.py --num_samples {args.num_samples}"
        success, output = run_command(cmd)
        if not success:
            return
        
        # 2. 训练模型
        logger.info("步骤 2/3: 训练模型")
        cmd = "python scripts/train_model.py"
        if args.no_lora:
            cmd += " --no_lora"
        success, output = run_command(cmd)
        if not success:
            return
        
        # 3. 运行推理
        logger.info("步骤 3/3: 运行推理演示")
        cmd = "python scripts/inference.py --local_model"
        success, output = run_command(cmd)
        if not success:
            return
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()