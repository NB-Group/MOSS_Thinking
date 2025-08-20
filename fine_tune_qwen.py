import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def fine_tune_qwen():
    """
    使用 QLoRA 在本地微调 Qwen-1.5-1.8B-Chat 模型。
    """
    # --- 1. 配置 ---
    # 模型和数据集路径
    model_id = "Qwen/Qwen1.5-1.8B-Chat"
    dataset_file = "qwen_thinking_distill_data_large.jsonl"
    
    # 微调后新模型的名称
    new_model_name = "qwen1.5-1.8b-thinking-chat-lora"

    # QLoRA 和 BitsAndBytes 配置
    use_4bit = True
    bnb_4bit_compute_dtype = "bfloat16" # 如果您的 GPU 不支持 bfloat16，请改为 "float16"
    bnb_4bit_quant_type = "nf4"
    
    # LoRA 配置
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    
    # 训练参数
    num_train_epochs = 1
    batch_size = 2 # 根据您的 VRAM 大小调整
    gradient_accumulation_steps = 4 # 梯度累积
    learning_rate = 2e-4
    max_seq_length = 2048 # 根据您的 VRAM 和数据调整

    # --- 2. 加载数据集 ---
    print("正在加载数据集...")
    # 加载 JSONL 文件
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    # --- 3. 加载模型和分词器 ---
    print(f"正在加载模型: {model_id}")
    
    # 为 4-bit 量化模型配置 BitsAndBytes
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # 自动将模型分配到可用设备 (如 GPU)
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("模型和分词器加载完毕。")

    # --- 4. 配置 LoRA ---
    print("正在配置 LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[ # 针对 Qwen1.5 的注意力模块
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # --- 5. 配置训练参数 ---
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # 如果您的 GPU 支持 bf16，设为 True 以获得更好性能
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # --- 6. 定义格式化函数 ---
    # 这个函数将每个数据点（一个包含 'instruction', 'output' 的字典）转换为一个格式化的字符串
    def formatting_func(example):
        # 构建符合 Qwen1.5 聊天模板的 messages 列表
        # instruction 是用户的问题，output 是模型的回答
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']}
        ]
        
        # 应用 Qwen1.5 的聊天模板，生成用于训练的单个字符串
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # --- 7. 初始化 SFTTrainer ---
    print("正在初始化训练器...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,      # 使用格式化函数
        args=training_arguments,
    )

    # --- 8. 开始训练 ---
    print("开始模型微调...")
    trainer.train()
    print("模型微调完成！")

    # --- 9. 保存训练好的 LoRA 适配器 ---
    print(f"正在将训练好的 LoRA 适配器保存到 ./{new_model_name}")
    trainer.model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print("所有内容已成功保存。")

if __name__ == "__main__":
    fine_tune_qwen()
