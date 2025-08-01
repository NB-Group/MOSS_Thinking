from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型路径（可以是本地目录或huggingface上的ID）
model_path = "你的模型路径"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",                # 自动分配GPU/CPU
    torch_dtype=torch.float16,        # 内存不够可试用 torch.float32 或 cpu
    low_cpu_mem_usage=True,
    offload_folder="./offload",      # 如果内存小推荐添加
    trust_remote_code=True
)
model.eval()

# 构造聊天内容
messages = [
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "我是一个AI助手"},
    {"role": "user", "content": "你能做什么？"}
]

# 使用 chat_template 拼接输入
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

# 解码并截取新内容
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("助手：", response.strip())
