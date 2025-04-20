"""
模型训练脚本 - 实现知识蒸馏过程
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.data_processing import load_processed_dataset
from utils.model_utils import load_tokenizer_and_model, prepare_training_data, create_trainer
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
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
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用模型路径或从ModelScope下载
    model_path = args.model_path
    if model_path is None:
        logger.info(f"从ModelScope下载学生模型: {config.STUDENT_MODEL_ID}")
        from utils.data_processing import download_models
        _, model_path = download_models()
    
    # 加载分词器和模型
    logger.info("加载分词器和模型")
    tokenizer, model = load_tokenizer_and_model(
        model_path=model_path,
        use_lora=not args.no_lora
    )
    
    # 加载处理后的数据集
    logger.info(f"加载数据集: {args.data_dir}")
    dataset = load_processed_dataset(args.data_dir)
    
    # 准备训练数据
    logger.info("准备训练数据")
    processed_datasets = prepare_training_data(tokenizer, dataset)
    
    # 创建训练器
    logger.info("创建训练器")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        output_dir=args.output_dir
    )
    
    # 开始训练
    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存模型和分词器
    logger.info("保存模型")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练状态
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info(f"训练完成，模型已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()