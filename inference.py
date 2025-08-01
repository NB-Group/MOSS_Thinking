import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import os

def run_inference():
    """
    加载使用 LoRA 微调后的 Qwen-1.5 模型并进行交互式推理。
    """
    # --- 1. 配置 ---
    # 基础模型 ID
    base_model_id = "G:\Models\MOSS_Thinking"
    # LoRA 适配器路径 (我们刚刚训练好的)
    # lora_adapter_path = "G:\Models\MOSS_Thinking"
    
    print("正在加载分词器...")
    # --- 2. 加载分词器 ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("正在加载基础模型...")
    # --- 3. 加载基础模型 ---
    # 注意：这里我们不需要 BitsAndBytesConfig，因为我们是在推理
    # 如果你的显存足够，可以直接加载 FP16/BF16 模型以获得更快的速度
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16, # 或者 torch.float16
        device_map="auto",
        trust_remote_code=True
    )
    
    # print(f"正在加载 LoRA 适配器: {lora_adapter_path}")
    # --- 4. 加载并融合 LoRA 适配器 ---
    # 这会将 LoRA 权重应用到基础模型上
    # model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    model.eval() # 设置为评估模式
    print("-" * 30)

    # --- 5. 对话循环 ---
    # 初始化 TextStreamer 用于流式输出
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = []
    while True:
        try:
            user_input = input(">>")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # 将用户输入和历史记录格式化为聊天模板
            messages = []
            for role, content in history:
                 messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_input})

            # 应用模板
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 分词
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            # 生成回复 (使用 streamer)
            generated_ids = model.generate(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            
            # 解码完整的响应以用于历史记录
            response_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

            # streamer 会自动打印输出，所以我们不再需要下面的 print 语句
            # print(f"{response}")
            
            # 更新历史记录
            history.append(("user", user_input))
            history.append(("assistant", response))

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_inference()
