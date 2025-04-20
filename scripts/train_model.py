"""
模型训练脚本 - 实现知识蒸馏过程
"""

import os
import argparse
import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.data_processing import load_processed_dataset
from utils.model_utils import load_tokenizer_and_model, prepare_training_data, create_trainer
import config

# 设置更详细的日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_model_cache(model_id):
    """
    检查模型是否已经在缓存中
    
    Args:
        model_id: 模型ID，格式为 "组织/模型名称" 
        
    Returns:
        str: 缓存路径，如果存在；None，如果不存在
    """
    logger.info(f"检查模型缓存: {model_id}")
    
    # 检查常见的缓存路径
    # 1. ModelScope默认缓存路径
    home_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", model_id.replace("/", "_"))
    logger.info(f"检查路径1: {home_cache}")
    
    # 2. 工作目录缓存路径
    work_cache = os.path.join(config.CACHE_DIR, model_id.replace("/", "_"))
    logger.info(f"检查路径2: {work_cache}")
    
    # 3. 工作目录缓存路径(第二种格式) 
    work_cache2 = os.path.join(config.CACHE_DIR, model_id.split("/")[0], model_id.split("/")[1])
    logger.info(f"检查路径3: {work_cache2}")
    
    # 4. 工作目录缓存路径(第三种格式)
    work_cache3 = os.path.join(config.CACHE_DIR, model_id.split("/")[0], model_id.split("/")[1].replace(".", "___"))
    logger.info(f"检查路径4: {work_cache3}")
    
    # 检查这些路径是否存在
    for cache_path in [home_cache, work_cache, work_cache2, work_cache3]:
        if os.path.exists(cache_path):
            logger.info(f"在缓存中找到模型: {cache_path}")
            return cache_path
            
    logger.info(f"未在缓存中找到模型: {model_id}")
    return None

def main():
    logger.info("训练脚本启动")
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="训练能够'边想边说'的模型")
    parser.add_argument("--model_path", type=str, default=None,
                      help="学生模型的路径，默认使用配置文件中指定的模型")
    parser.add_argument("--data_dir", type=str, default=config.PROCESSED_DATA_DIR,
                      help=f"处理后的数据目录，默认为{config.PROCESSED_DATA_DIR}")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                      help=f"输出目录，默认为{config.OUTPUT_DIR}")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                      help=f"批次大小，默认为{config.BATCH_SIZE}")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                      help=f"学习率，默认为{config.LEARNING_RATE}")
    parser.add_argument("--epochs", type=int, default=config.NUM_TRAIN_EPOCHS,
                      help=f"训练轮数，默认为{config.NUM_TRAIN_EPOCHS}")
    parser.add_argument("--no_lora", action="store_true", 
                      help="不使用LoRA进行微调，默认使用")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="从检查点恢复训练")
    
    logger.info("解析命令行参数")
    args = parser.parse_args()
    logger.info(f"参数解析完成: data_dir={args.data_dir}, output_dir={args.output_dir}, no_lora={args.no_lora}")
    
    # 创建输出目录
    logger.info(f"创建输出目录: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用模型路径或从缓存/ModelScope获取
    model_path = args.model_path
    logger.info(f"指定的模型路径: {model_path}")
    
    if model_path is None:
        logger.info("模型路径未指定，将使用或下载默认模型")
        # 首先检查缓存
        cached_path = check_model_cache(config.STUDENT_MODEL_ID)
        if (cached_path):
            logger.info(f"使用缓存中的学生模型: {cached_path}")
            model_path = cached_path
        else:
            # 如果缓存中没有，则下载
            logger.info(f"从ModelScope下载学生模型: {config.STUDENT_MODEL_ID}")
            try:
                # 只下载学生模型，不下载教师模型
                logger.info("尝试使用snapshot_download下载模型")
                from modelscope import snapshot_download
                model_path = snapshot_download(config.STUDENT_MODEL_ID, cache_dir=config.CACHE_DIR)
                logger.info(f"学生模型下载完成: {model_path}")
            except Exception as e:
                logger.warning(f"使用snapshot_download下载失败: {str(e)}")
                logger.warning(f"尝试使用download_models方法下载")
                from utils.data_processing import download_models
                _, model_path = download_models()
    
    logger.info(f"最终确定的模型路径: {model_path}")
    
    # 检查数据目录
    logger.info(f"检查数据目录: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 检查数据文件
    train_file = os.path.join(args.data_dir, "train.json")
    val_file = os.path.join(args.data_dir, "val.json")
    
    logger.info(f"检查训练数据文件: {train_file}")
    if not os.path.exists(train_file):
        logger.error(f"训练数据文件不存在: {train_file}")
        sys.exit(1)
        
    logger.info(f"检查验证数据文件: {val_file}")
    if not os.path.exists(val_file):
        logger.error(f"验证数据文件不存在: {val_file}")
        sys.exit(1)
    
    # 加载分词器和模型
    logger.info(f"开始加载分词器...")
    t0 = time.time()
    
    try:
        logger.info(f"尝试加载分词器和模型：{model_path}")
        tokenizer, model = load_tokenizer_and_model(
            model_path=model_path,
            use_lora=not args.no_lora
        )
        logger.info(f"分词器和模型加载完成，耗时 {time.time() - t0:.2f} 秒")
    except Exception as e:
        logger.error(f"加载分词器和模型失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 加载处理后的数据集
    logger.info(f"加载数据集: {args.data_dir}")
    t0 = time.time()
    
    try:
        dataset = load_processed_dataset(args.data_dir)
        logger.info(f"数据集加载完成，耗时 {time.time() - t0:.2f} 秒")
        logger.info(f"训练集大小: {len(dataset['train'])}，验证集大小: {len(dataset['validation'])}")
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 准备训练数据
    logger.info("准备训练数据...")
    t0 = time.time()
    
    try:
        processed_datasets = prepare_training_data(tokenizer, dataset)
        logger.info(f"训练数据准备完成，耗时 {time.time() - t0:.2f} 秒")
        logger.info(f"处理后训练集大小: {len(processed_datasets['train'])}，验证集大小: {len(processed_datasets['validation'])}")
    except Exception as e:
        logger.error(f"准备训练数据失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 创建训练器
    logger.info("创建训练器...")
    t0 = time.time()
    
    try:
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["validation"],
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs
        )
        logger.info(f"训练器创建完成，耗时 {time.time() - t0:.2f} 秒")
    except Exception as e:
        logger.error(f"创建训练器失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 开始训练
    logger.info("开始训练...")
    train_start_time = time.time()
    
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logger.info(f"训练完成，总耗时 {time.time() - train_start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"训练过程失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 保存模型和分词器
    logger.info("保存模型和分词器...")
    t0 = time.time()
    
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"模型和分词器保存完成，耗时 {time.time() - t0:.2f} 秒")
    except Exception as e:
        logger.error(f"保存模型和分词器失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 保存训练状态
    logger.info("保存训练状态和指标...")
    
    try:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"训练状态和指标保存完成")
    except Exception as e:
        logger.error(f"保存训练状态和指标失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    total_duration = time.time() - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    logger.info(f"训练流程全部完成，总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
    logger.info(f"模型已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()