"""
模型工具函数
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokenizer_and_model(model_path: str, use_lora: bool = config.USE_LORA) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    加载分词器和模型
    
    Args:
        model_path: 模型路径
        use_lora: 是否使用LoRA微调
        
    Returns:
        tokenizer, model 元组
    """
    logger.info(f"从 {model_path} 加载分词器")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 确保分词器中包含特殊标记
    special_tokens_dict = {}
    if config.THINK_START_TOKEN not in tokenizer.get_vocab():
        special_tokens_dict["additional_special_tokens"] = [config.THINK_START_TOKEN, config.THINK_END_TOKEN]
    
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    
    logger.info(f"从 {model_path} 加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # 如果有特殊标记，调整模型的嵌入大小
    if special_tokens_dict:
        model.resize_token_embeddings(len(tokenizer))
    
    # 配置LoRA
    if use_lora:
        logger.info("配置LoRA微调")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=config.TARGET_MODULES
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return tokenizer, model

def prepare_training_data(tokenizer: AutoTokenizer, dataset: Dataset) -> Dataset:
    """
    处理数据集为训练格式
    
    Args:
        tokenizer: 分词器
        dataset: 数据集
        
    Returns:
        处理后的数据集
    """
    logger.info("准备训练数据")
    
    def process_function(examples):
        # 构建输入文本
        inputs = [f"用户: {ex}" for ex in examples["input"]]
        targets = [f"助手: {ex}" for ex in examples["output"]]
        
        # 编码输入
        model_inputs = tokenizer(
            inputs,
            max_length=config.MAX_SOURCE_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 编码目标
        labels = tokenizer(
            targets,
            max_length=config.MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # 将填充的token标记为-100，使得在计算损失时被忽略
        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
        model_inputs["labels"] = labels
        
        return model_inputs
    
    processed_datasets = dataset.map(
        process_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="处理数据集"
    )
    
    return processed_datasets

def create_trainer(model: AutoModelForCausalLM, 
                  tokenizer: AutoTokenizer, 
                  train_dataset: Dataset, 
                  eval_dataset: Optional[Dataset] = None,
                  output_dir: str = config.OUTPUT_DIR) -> Trainer:
    """
    创建训练器
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        output_dir: 输出目录
        
    Returns:
        Trainer实例
    """
    logger.info("创建训练器")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=500 if eval_dataset is not None else None,
        save_steps=500,
        save_total_limit=3,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        logging_steps=config.LOGGING_STEPS,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt"
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    return trainer