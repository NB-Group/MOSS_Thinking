"""
下载模型和数据集脚本 - 仅从ModelScope下载所需资源（无需访问Huggingface）
"""

import os
import argparse
import logging
import sys
import subprocess

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 只使用ModelScope的功能
from modelscope import snapshot_download
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="从ModelScope下载模型和数据集")
    parser.add_argument("--teacher_model_id", type=str, default=config.TEACHER_MODEL_ID,
                       help=f"教师模型ID，默认为{config.TEACHER_MODEL_ID}")
    parser.add_argument("--student_model_id", type=str, default=config.STUDENT_MODEL_ID,
                       help=f"学生模型ID，默认为{config.STUDENT_MODEL_ID}")
    parser.add_argument("--dataset_id", type=str, default=config.DATASET_ID,
                       help=f"数据集ID，默认为{config.DATASET_ID}")
    parser.add_argument("--cache_dir", type=str, default=config.CACHE_DIR,
                       help=f"缓存目录，默认为{config.CACHE_DIR}")
    parser.add_argument("--download_student", action="store_true", 
                       help="是否下载学生模型")
    parser.add_argument("--download_teacher", action="store_true", 
                       help="是否下载教师模型")
    parser.add_argument("--download_dataset", action="store_true", 
                       help="是否下载数据集")
    parser.add_argument("--download_all", action="store_true", 
                       help="下载所有资源")
    
    args = parser.parse_args()
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 下载教师模型
    if args.download_teacher or args.download_all:
        logger.info(f"正在下载教师模型: {args.teacher_model_id}")
        try:
            teacher_path = snapshot_download(args.teacher_model_id, cache_dir=args.cache_dir)
            logger.info(f"教师模型下载完成，保存在: {teacher_path}")
        except Exception as e:
            logger.error(f"教师模型下载失败: {e}")
            logger.info("您可以尝试手动下载教师模型，从ModelScope网站下载后放入缓存目录")
    
    # 下载学生模型
    if args.download_student or args.download_all:
        logger.info(f"正在下载学生模型: {args.student_model_id}")
        try:
            student_path = snapshot_download(args.student_model_id, cache_dir=args.cache_dir)
            logger.info(f"学生模型下载完成，保存在: {student_path}")
        except Exception as e:
            logger.error(f"学生模型下载失败: {e}")
            logger.info("您可以尝试手动下载学生模型，从ModelScope网站下载后放入缓存目录")
    
    # 下载数据集，只使用ModelScope的方法
    if args.download_dataset or args.download_all:
        logger.info(f"正在下载数据集: {args.dataset_id}")
        try:
            # 尝试使用modelscope CLI工具下载数据集
            logger.info("尝试使用modelscope CLI工具下载数据集...")
            cmd = f"python -m modelscope.cli.download_dataset --dataset_id {args.dataset_id} --cache_dir {args.cache_dir}"
            logger.info(f"执行命令: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"数据集下载完成: {result.stdout}")
                download_success = True
            else:
                logger.warning(f"使用CLI下载失败: {result.stderr}")
                download_success = False
            
            # 如果CLI方法失败，给出手动下载指导
            if not download_success:
                logger.warning("自动下载数据集失败，请手动下载")
                logger.info("手动下载步骤:")
                logger.info(f"1. 访问 https://www.modelscope.cn/datasets/{args.dataset_id}/summary")
                logger.info("2. 点击'下载'按钮下载数据集")
                logger.info(f"3. 将数据集解压到 {args.cache_dir} 目录")
        except Exception as e:
            logger.error(f"数据集下载失败: {e}")
            logger.info("请从ModelScope网站手动下载数据集")
    
    if not any([args.download_teacher, args.download_student, args.download_dataset, args.download_all]):
        logger.warning("没有指定要下载的资源，请使用 --download_teacher, --download_student, --download_dataset 或 --download_all")
        parser.print_help()

if __name__ == "__main__":
    main()