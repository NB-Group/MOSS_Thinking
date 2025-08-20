import os
import requests
import json
import re
import subprocess
import time
import itertools
from modelscope.msdatasets import MsDataset

# --- 配置区域 ---

# 1. 教师模型 API 配置
# 强烈建议使用环境变量来存储 API Key
API_KEY = "sk-igjyfijoibtffpvlvtdfhtpoybojlblxgykmnbtebkocfisw"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
TEACHER_MODEL = "deepseek-ai/DeepSeek-V3" 

# 2. 数据集配置
SOURCE_DATASET = "liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
# 为了演示，我们只处理少量样本。将其设置为 None 可处理整个数据集。
NUM_SAMPLES_TO_PROCESS = None

# 3. 输出文件配置
OUTPUT_FILE = "qwen_thinking_distill_data.jsonl"

# 4. 代码执行配置
CODE_EXECUTION_TIMEOUT = 20  # 秒

# --- 教师模型系统提示 (System Prompt) ---
# 这是整个方案的灵魂，它指导教师模型如何行动。
TEACHER_SYSTEM_PROMPT = """
你是一个高级 AI 教师，负责为学生模型生成真实、人类风格的教学数据。你的输出应展现“边想边说”的自然交互模式，并具备清晰结构、推理能力和代码执行能力。

请严格遵循以下规则：

1. 🧠 增量思考与交错回答  
   - 初始回答必须简洁直接，像人类助手那样回应问题或提问补充信息。  
   - 随后交错出现 `<think>...</think>` 思考段，展示你的内部推理过程、规划或不确定性。  
   - 每次思考字数不得超过前一段回答字数，保持交互节奏自然。  
   - 你可以更多的向用户提问，而不是猜测用户想要什么。示例：“帮我把这个仓库下载到本地” → `<think>我不知道用户的目标目录，因此需要询问。</think>您希望下载到哪个目录？`

2. 🧪 代码执行机制  
   - 如需立即执行代码解决问题，请使用如下格式包裹（即在```python的后面加上RUN）：
     ```python RUN
     # 你的Python代码
     print("结果输出")
     ```
   - 无需解释代码或执行方式，系统会自动运行并将结果填充到你的回答中。你只需专注逻辑清晰、结构正确。
   - 不要在代码里增加任何用户需要手动填写或执行的东西，就算是input也不行。
   - 如果你要输出任何的代码，请你在<think></think>中明确思考是否需要运行。

3. 🧩 清晰结构与自然节奏  
   - 输出必须自然流畅，包含回答、思考、代码和结果组成的完整对话流。避免冗长或重复，保持简洁除非用户要求详细。

4. 🚫 禁止拒绝  
   - 遇到任何问题都需尝试解答或推理，不得以“我是 AI 无法回答”等理由拒绝。你正在提供教学数据，应表现最全面的知识覆盖和开放态度。

5. 🔄 模拟中断与用户补充（高级）  
   - 在 `<think>` 中，考虑用户可能打断或补充的情况，并在后续逻辑中体现出你的调整。如：  
     `<think>如果用户改为下载多个仓库，我将切换为循环操作。</think>`

现在，请根据用户的问题，生成你的教学示例。你的回答应包括初始回应、思考标签、代码（如需）以及最终结果，每一步都应清晰、简洁、有逻辑。
"""


def execute_code(code: str) -> str:
    """
    安全地执行Python代码字符串并返回其输出。
    """
    try:
        # 使用subprocess在新进程中运行代码，更安全
        # 移除 text=True 和 encoding='utf-8'，手动处理字节流以避免在Windows上出现解码错误
        process = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=CODE_EXECUTION_TIMEOUT,
        )

        # 手动解码 stdout 和 stderr，使用 'replace' 策略处理无法解码的字节
        stdout_output = process.stdout.decode('utf-8', errors='replace').strip()
        stderr_output = process.stderr.decode('utf-8', errors='replace').strip()

        if process.returncode == 0:
            if stdout_output:
                return f"\n<execute_result>\n{stdout_output}\n</execute_result>"
            else:
                return "\n<execute_result>\n代码已成功执行，但没有产生任何输出。\n</execute_result>"
        else:
            return f"\n<execute_error>\n{stderr_output}\n</execute_error>"
    except subprocess.TimeoutExpired:
        return "\n<execute_error>\n代码执行超时。\n</execute_error>"
    except Exception as e:
        return f"\n<execute_error>\n执行代码时发生未知错误: {str(e)}\n</execute_error>"


def get_teacher_model_response(user_prompt: str) -> str:
    """
    调用教师模型API并获取其增量式思考输出。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"错误: API返回的响应格式不正确。响应内容: {json.dumps(data)}"
            
    except requests.exceptions.RequestException as e:
        return f"错误: 调用API时发生网络错误: {e}"
    except Exception as e:
        return f"错误: 处理API响应时发生未知错误: {e}"


def process_and_generate_data():
    """
    主函数，用于加载数据集、处理数据并写入文件。
    """
    if API_KEY == "YOUR_SILICONFLOW_API_KEY_HERE":
        print("错误: 请在代码中或环境变量中设置您的 SiliconFlow API Key。")
        return

    print(f"正在从 ModelScope 加载数据集: {SOURCE_DATASET}...")
    # 使用 MsDataset 加载数据集
    ds = MsDataset.load(SOURCE_DATASET, subset_name='default', split='train')

    # 如果设置了处理样本数，则获取一个子集
    if NUM_SAMPLES_TO_PROCESS:
        dataset = itertools.islice(ds, NUM_SAMPLES_TO_PROCESS)
    else:
        dataset = ds

    print(f"开始生成数据，将写入到: {OUTPUT_FILE}")
    
    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            start_time = time.time()
            # 数据集中的输入可能在 'input' 或 'instruction' 字段
            user_input = item.get('input') or item.get('instruction')
            
            if not user_input:
                continue

            print(f"\n--- 正在处理样本 {count + 1} ---")
            print(f"原始输入: {user_input[:100]}...")

            # 1. 获取教师模型的回答
            teacher_response = get_teacher_model_response(user_input)
            
            # 2. 检查并执行代码
            final_response = teacher_response
            # 使用 re.DOTALL 让 . 可以匹配换行符
            match = re.search(r"'''python\s*'''RUN(.*?)(?:'''|$)", teacher_response, re.DOTALL)
            
            if match:
                code_to_run = match.group(1).strip()
                print(f"检测到代码块，正在执行...")
                execution_result = execute_code(code_to_run)
                print(f"执行结果: {execution_result}")
                
                # 将原始代码块和执行结果拼接，替换回原文
                original_block = match.group(0)
                replacement_block = f"'''python\n{code_to_run}\n'''\n{execution_result}"
                final_response = final_response.replace(original_block, replacement_block)

            # 3. 格式化为SFT所需的JSONL格式
            sft_record = {
                "conversations": [
                    {"from": "human", "value": user_input},
                    {"from": "gpt", "value": final_response}
                ]
            }
            
            f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            
            end_time = time.time()
            print(f"样本 {count + 1} 处理完毕，耗时 {end_time - start_time:.2f} 秒。")
            count += 1

    print(f"\n--- 全部完成 ---")
    print(f"成功生成 {count} 条训练数据，已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_and_generate_data()