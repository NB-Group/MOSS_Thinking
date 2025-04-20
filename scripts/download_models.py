"""
下载模型和数据集脚本 - 仅从ModelScope下载所需资源（无需访问Huggingface）
"""

import os
import argparse
import logging
import sys
import subprocess
import requests
import time

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

def download_file(url, save_path):
    """
    下载文件并显示进度
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        logger.info(f"开始下载文件: {url}")
        logger.info(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 下载文件
        downloaded = 0
        start_time = time.time()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 计算下载进度和速度
                    percent = downloaded / total_size * 100
                    elapsed_time = time.time() - start_time
                    speed = downloaded / (1024 * 1024) / elapsed_time if elapsed_time > 0 else 0
                    
                    # 显示进度
                    logger.info(f"下载进度: {percent:.2f}% ({downloaded/(1024*1024):.2f}MB/{total_size/(1024*1024):.2f}MB), 速度: {speed:.2f} MB/s")
        
        logger.info(f"文件下载完成，保存在: {save_path}")
        return True
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return False

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
    parser.add_argument("--dataset_url", type=str, default=None,
                       help="数据集直接下载链接")
    
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
    
    # 下载数据集
    if args.download_dataset or args.download_all:
        logger.info(f"正在下载数据集: {args.dataset_id}")
        dataset_downloaded = False
        
        # 首先尝试使用MsDataset.load直接下载
        try:
            logger.info("使用MsDataset.load方法直接下载数据集...")
            from modelscope.msdatasets import MsDataset
            # 使用固定的subset_name和split值来简化操作
            dataset = MsDataset.load(args.dataset_id, subset_name='default', split='train', cache_dir=args.cache_dir)
            logger.info(f"数据集下载完成!")
            # 显示数据集的一些基本信息
            logger.info(f"数据集大小: {len(dataset)} 条样本")
            logger.info(f"数据集示例: {dataset[0] if len(dataset) > 0 else 'empty'}")
            dataset_downloaded = True
        except ImportError as e:
            logger.warning(f"导入MsDataset失败: {e}")
        except Exception as e:
            logger.warning(f"使用MsDataset下载失败: {e}")
        
        # 尝试使用直接URL下载
        if not dataset_downloaded and args.dataset_url:
            logger.info(f"使用提供的URL下载数据集: {args.dataset_url}")
            dataset_filename = os.path.basename(args.dataset_url)
            dataset_save_path = os.path.join(args.cache_dir, dataset_filename)
            dataset_downloaded = download_file(args.dataset_url, dataset_save_path)
            
            if dataset_downloaded:
                # 解压数据集（如果是压缩文件）
                if dataset_filename.endswith(('.zip', '.tar.gz', '.tgz')):
                    try:
                        import tarfile
                        import zipfile
                        
                        logger.info(f"尝试解压数据集: {dataset_save_path}")
                        extract_dir = os.path.join(args.cache_dir, args.dataset_id.replace('/', '-'))
                        os.makedirs(extract_dir, exist_ok=True)
                        
                        if dataset_filename.endswith('.zip'):
                            with zipfile.ZipFile(dataset_save_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                        elif dataset_filename.endswith(('.tar.gz', '.tgz')):
                            with tarfile.open(dataset_save_path, 'r:gz') as tar_ref:
                                tar_ref.extractall(extract_dir)
                                
                        logger.info(f"数据集解压完成，保存在: {extract_dir}")
                    except Exception as e:
                        logger.error(f"解压数据集失败: {e}")
                        logger.info(f"请手动解压数据集 {dataset_save_path} 到目录 {extract_dir}")
        
        # 如果所有方法都失败，给出手动下载指导
        if not dataset_downloaded:
            logger.warning("自动下载数据集失败，请按以下方式手动下载")
            logger.info("手动下载指南:")
            logger.info(f"方法1: 使用Python代码下载")
            logger.info(f"  from modelscope.msdatasets import MsDataset")
            logger.info(f"  ds = MsDataset.load('{args.dataset_id}', subset_name='default', split='train')")
            logger.info(f"方法2: 通过网站下载")
            logger.info(f"  1. 访问 https://www.modelscope.cn/datasets/{args.dataset_id}/summary")
            logger.info(f"  2. 点击'下载'按钮下载数据集")
            logger.info(f"  3. 将数据集解压到 {os.path.join(args.cache_dir, args.dataset_id.replace('/', '-'))} 目录")
    
    if not any([args.download_teacher, args.download_student, args.download_dataset, args.download_all]):
        logger.warning("没有指定要下载的资源，请使用 --download_teacher, --download_student, --download_dataset 或 --download_all")
        parser.print_help()

if __name__ == "__main__":
    main()