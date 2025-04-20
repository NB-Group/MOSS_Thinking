"""
数据处理工具
"""

import os
import json
import torch
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
# 只使用ModelScope的API，不使用HuggingFace的datasets
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
# 使用MsDataset直接下载数据集
from modelscope.msdatasets import MsDataset
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset() -> Any:
    """
    从ModelScope下载数据集
    """
    logger.info(f"正在下载数据集: {config.DATASET_ID}")
    try:
        # 直接使用MsDataset.load下载数据集
        logger.info(f"使用MsDataset.load下载数据集: {config.DATASET_ID}")
        dataset = MsDataset.load(config.DATASET_ID, subset_name='default', split='train', cache_dir=config.CACHE_DIR)
        logger.info(f"数据集下载完成，包含 {len(dataset)} 条样本")
        
        # 根据样本结构转换数据集
        # 样本结构: {'instruction': '...', 'input': '...', 'output': '...', 'repo_name': '...', 
        #           'prompt_tokens_len': int, 'reasoning_content_tokens_len': int, 
        #           'content_tokens_len': int, 'score': int}
        dataset_dict = {"train": []}
        
        # 遍历数据集样本
        for item in dataset:
            # 确保样本包含必要的字段
            if "instruction" in item and "output" in item:
                # 构建输入文本，结合instruction和input字段
                input_text = item["instruction"]
                if "input" in item and item["input"]:
                    input_text += "\n" + item["input"]
                
                # 将数据转换为所需格式
                dataset_dict["train"].append({
                    "input": input_text,
                    "output": item["output"]  # 输出已经包含了<think>标记的思考过程
                })
        
        logger.info(f"已转换 {len(dataset_dict['train'])} 条数据")
        return dataset_dict
    except Exception as e:
        logger.error(f"数据集下载失败: {e}")
        logger.info("请尝试手动下载数据集并放入缓存目录")
        logger.info("手动下载方法:")
        logger.info("from modelscope.msdatasets import MsDataset")
        logger.info(f"ds = MsDataset.load('{config.DATASET_ID}', subset_name='default', split='train')")
        # 返回空数据集，让流程能继续
        return {"train": []}

def load_dataset_from_files(dataset_path: str) -> Dict:
    """从本地文件加载数据集"""
    logger.info(f"从目录加载数据集: {dataset_path}")
    
    # 搜索目录下的json文件
    json_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json') or file.endswith('.jsonl'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        logger.warning(f"在 {dataset_path} 中没有找到json文件")
        return {"train": []}
    
    # 加载每个json文件的内容
    all_data = []
    for json_file in json_files:
        logger.info(f"加载文件: {json_file}")
        try:
            if json_file.endswith('.jsonl'):
                # jsonl格式
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_data.append(json.loads(line))
            else:
                # json格式
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
        except Exception as e:
            logger.error(f"加载文件 {json_file} 失败: {e}")
    
    logger.info(f"已加载 {len(all_data)} 条数据")
    return {"train": all_data}

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

def generate_teacher_outputs(teacher_model_path: str, dataset: Dict, 
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
    samples = dataset["train"]
    if num_samples is not None:
        samples = samples[:min(num_samples, len(samples))]
    
    logger.info(f"使用教师模型生成 {len(samples)} 条样本的思考过程")
    
    for idx, sample in enumerate(tqdm(samples)):
        # 准备输入
        question = sample.get("input", sample.get("prompt", sample.get("question", "")))
        if not question:
            logger.warning(f"样本 {idx} 没有有效的输入字段，跳过")
            continue
            
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

def load_processed_dataset(data_dir: str = config.PROCESSED_DATA_DIR) -> Dict:
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
    
    return {
        "train": train_data,
        "validation": val_data
    }