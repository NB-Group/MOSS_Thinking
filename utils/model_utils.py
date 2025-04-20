"""
模型工具函数
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
# 使用ModelScope中的AutoModelForCausalLM和AutoTokenizer来替代Huggingface的对应组件
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import (
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
# 不再直接依赖datasets库
import torch.utils.data

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(torch.utils.data.Dataset):
    """简单的数据集类，替代Huggingface的Dataset"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

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

def prepare_training_data(tokenizer: AutoTokenizer, dataset: Dict) -> Dict[str, SimpleDataset]:
    """
    处理数据集为训练格式
    
    Args:
        tokenizer: 分词器
        dataset: 数据集字典，包含train和validation
        
    Returns:
        处理后的数据集
    """
    logger.info("准备训练数据")
    
    def process_batch(batch_data):
        """批处理函数"""
        inputs = [f"用户: {item['input']}" for item in batch_data]
        targets = [f"助手: {item['output']}" for item in batch_data]
        
        # 编码输入
        model_inputs_dict = tokenizer(
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
        model_inputs_dict["labels"] = labels
        
        return model_inputs_dict
    
    # 处理批次并转换为Dataset格式
    def process_dataset(data_list):
        batch_size = 8  # 内部处理批次大小
        processed_data = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_inputs = process_batch(batch)
            
            # 将批次数据拆分为单个样本
            for j in range(len(batch)):
                item = {}
                for k, v in batch_inputs.items():
                    item[k] = v[j]
                processed_data.append(item)
                
        return SimpleDataset(processed_data)
    
    # 处理训练集和验证集
    logger.info(f"处理 {len(dataset['train'])} 条训练数据")
    train_dataset = process_dataset(dataset["train"])
    
    logger.info(f"处理 {len(dataset['validation'])} 条验证数据")
    eval_dataset = process_dataset(dataset["validation"])
    
    return {
        "train": train_dataset,
        "validation": eval_dataset
    }

def create_trainer(model: AutoModelForCausalLM, 
                  tokenizer: AutoTokenizer, 
                  train_dataset: SimpleDataset, 
                  eval_dataset: Optional[SimpleDataset] = None,
                  output_dir: str = config.OUTPUT_DIR,
                  batch_size: int = config.BATCH_SIZE,
                  learning_rate: float = config.LEARNING_RATE,
                  num_train_epochs: int = config.NUM_TRAIN_EPOCHS) -> Trainer:
    """
    创建训练器
    
    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        output_dir: 输出目录
        batch_size: 训练批次大小
        learning_rate: 学习率
        num_train_epochs: 训练轮数
        
    Returns:
        Trainer实例
    """
    logger.info("创建训练器")
    
    # 设置混合精度训练
    # 优先使用BF16（更快更精确），如果不支持则使用FP16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    
    logger.info(f"混合精度训练配置: bf16={use_bf16}, fp16={use_fp16}")
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=500 if eval_dataset is not None else None,
        save_steps=500,
        save_total_limit=3,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=num_train_epochs,
        warmup_ratio=config.WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        logging_steps=config.LOGGING_STEPS,
        fp16=use_fp16,  # 只有在不支持bf16时才使用fp16
        bf16=use_bf16,  # 优先使用bf16
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