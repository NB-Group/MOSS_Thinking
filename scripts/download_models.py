"""
下载模型和数据集脚本 - 从ModelScope下载所需资源
"""

import os
import argparse
import logging
import sys

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from modelscope import snapshot_download, MsDataset
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
        teacher_path = snapshot_download(args.teacher_model_id, cache_dir=args.cache_dir)
        logger.info(f"教师模型下载完成，保存在: {teacher_path}")
    
    # 下载学生模型
    if args.download_student or args.download_all:
        logger.info(f"正在下载学生模型: {args.student_model_id}")
        student_path = snapshot_download(args.student_model_id, cache_dir=args.cache_dir)
        logger.info(f"学生模型下载完成，保存在: {student_path}")
    
    # 下载数据集，使用ModelScope的MsDataset而不是HuggingFace的load_dataset
    if args.download_dataset or args.download_all:
        logger.info(f"正在下载数据集: {args.dataset_id}")
        try:
            # 使用ModelScope的API下载数据集
            dataset = MsDataset.load(args.dataset_id, cache_dir=args.cache_dir)
            logger.info(f"数据集下载完成，保存在: {args.cache_dir}")
            
            # 获取数据集大小信息
            if hasattr(dataset, "train") and hasattr(dataset.train, "__len__"):
                logger.info(f"数据集包含 {len(dataset.train)} 条训练数据")
            else:
                logger.info(f"数据集下载完成")
        except Exception as e:
            logger.error(f"数据集下载失败: {e}")
            logger.info("尝试使用HuggingFace datasets API下载数据集...")
            
            try:
                from datasets import load_dataset
                # 注意这里的格式，直接使用ModelScope的数据集ID
                dataset = load_dataset("modelscope/" + args.dataset_id, cache_dir=args.cache_dir)
                logger.info(f"数据集下载完成，包含 {len(dataset['train'])} 条训练数据")
            except Exception as e2:
                logger.error(f"所有尝试都失败: {e2}")
                logger.info("您可以尝试手动下载数据集，或者在训练时使用 --dataset_path 参数指定本地数据集")
    
    if not any([args.download_teacher, args.download_student, args.download_dataset, args.download_all]):
        logger.warning("没有指定要下载的资源，请使用 --download_teacher, --download_student, --download_dataset 或 --download_all")
        parser.print_help()

if __name__ == "__main__":
    main()