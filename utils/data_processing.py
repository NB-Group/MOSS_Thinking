"""
数据处理工具
"""

import os
import json
import torch
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers import AutoTokenizer as HFAutoTokenizer

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset() -> Dataset:
    """
    从ModelScope下载数据集
    """
    logger.info(f"正在下载数据集: {config.DATASET_ID}")
    dataset = load_dataset("modelscope:" + config.DATASET_ID, cache_dir=config.CACHE_DIR)
    logger.info(f"数据集下载完成，包含 {len(dataset['train'])} 条训练数据")
    return dataset

def download_models() -> Tuple[str, str]:
    """
    下载教师模型和学生模型，返回其本地路径
    """
    logger.info(f"正在下载教师模型: {config.TEACHER_MODEL_ID}")
    teacher_path = snapshot_download(config.TEACHER_MODEL_ID, cache_dir=config.CACHE_DIR)
    
    logger.info(f"正在下载学生模型: {config.STUDENT_MODEL_ID}")
    student_path = snapshot_download(config.STUDENT_MODEL_ID, cache_dir=config.CACHE_DIR)
    
    return teacher_path, student_path

def format_think_segments(text: str) -> str:
    """
    将教师模型的输出格式化为带有思考标记的文本
    
    教师模型的输出大致会是这样：
    Thought: 我需要计算斐波那契数列第10项
    Thought: 斐波那契数列定义是每项等于前两项之和
    Answer: 斐波那契数列第10项是55
    
    我们需要将其转换为:
    <think>我需要计算斐波那契数列第10项</think>
    <think>斐波那契数列定义是每项等于前两项之和</think>
    斐波那契数列第10项是55
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.startswith("Thought:"):
            thought_content = line[len("Thought:"):].strip()
            formatted_lines.append(f"{config.THINK_START_TOKEN} {thought_content} {config.THINK_END_TOKEN}")
        elif line.startswith("Answer:"):
            answer_content = line[len("Answer:"):].strip()
            formatted_lines.append(answer_content)
        elif "```" in line and "RUN" in line:
            # 处理带有运行标记的代码块
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def generate_teacher_outputs(teacher_model_path: str, dataset: Dataset, 
                            num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    使用教师模型生成带有思考步骤的输出
    
    Args:
        teacher_model_path: 教师模型路径
        dataset: 输入数据集
        num_samples: 处理的样本数量，None表示处理全部
        
    Returns:
        包含教师模型输出的数据集
    """
    logger.info("加载教师模型和分词器")
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
    
    # 准备教师模型的提示语模板
    teacher_prompt_template = """你是一个善于思考的AI助手。请对下面的问题一步一步思考并回答。
在回答过程中，请使用"Thought:"来表示你的思考过程，使用"Answer:"来表示最终答案。

用户问题：{question}

请先思考，再给出答案："""

    results = []
    samples = dataset["train"] if num_samples is None else dataset["train"].select(range(min(num_samples, len(dataset["train"]))))
    
    logger.info(f"使用教师模型生成 {len(samples)} 条样本的思考过程")
    
    for idx, sample in enumerate(tqdm(samples)):
        # 准备输入
        question = sample["input"]
        prompt = teacher_prompt_template.format(question=question)
        
        # 生成教师模型的输出
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_TARGET_LENGTH,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        teacher_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 格式化教师输出，添加思考标记
        formatted_output = format_think_segments(teacher_output)
        
        # 保存结果
        results.append({
            "input": question,
            "original_output": teacher_output,
            "formatted_output": formatted_output
        })
        
        if (idx + 1) % 10 == 0:
            logger.info(f"已处理 {idx + 1}/{len(samples)} 条样本")
    
    return results

def prepare_distillation_dataset(teacher_outputs: List[Dict[str, Any]], 
                               output_dir: str = config.PROCESSED_DATA_DIR) -> None:
    """
    准备用于蒸馏的数据集，保存为JSON文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = []
    val_data = []
    
    # 划分训练集和验证集
    val_size = min(config.VAL_SET_SIZE, int(len(teacher_outputs) * 0.1))
    
    for i, item in enumerate(teacher_outputs):
        entry = {
            "input": item["input"],
            "output": item["formatted_output"]
        }
        
        if i < val_size:
            val_data.append(entry)
        else:
            train_data.append(entry)
    
    # 保存为JSON文件
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存 {len(train_data)} 条训练数据和 {len(val_data)} 条验证数据到 {output_dir}")

def load_processed_dataset(data_dir: str = config.PROCESSED_DATA_DIR) -> DatasetDict:
    """
    加载处理后的数据集
    """
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"处理后的数据文件不存在: {train_path} 或 {val_path}")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })