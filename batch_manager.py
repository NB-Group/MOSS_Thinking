import os
import requests
import time
import json
import argparse

# --- 配置区域 ---
API_BASE_URL = "https://api.siliconflow.cn/v1"
# 默认的批量输入文件
DEFAULT_INPUT_FILE = "batch_input_for_siliconflow.jsonl"
# 默认的完整批量输入文件
DEFAULT_COMPLETE_INPUT_FILE = "batch_input_for_siliconflow_complete.jsonl"
# 最终生成的、用于微调的数据集文件
FINAL_TRAIN_DATASET_FILE = "qwen_thinking_distill_data_large.jsonl"

def get_api_key():
    """从环境变量中获取 API 密钥"""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("错误: 请设置 SILICONFLOW_API_KEY 环境变量。")
    return api_key

def upload_file(api_key, file_path):
    """上传文件到 SiliconFlow"""
    print(f"正在上传文件: {file_path}...")
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/jsonl")}
        data = {"purpose": "batch"}
        response = requests.post(f"{API_BASE_URL}/files", headers=headers, files=files, data=data)
    response.raise_for_status()
    file_object = response.json()
    print(f"文件上传成功。文件 ID: {file_object['id']}\n")
    return file_object

def create_batch(api_key, file_id):
    """使用上传的文件创建批量任务"""
    print(f"正在使用文件 ID 创建批量任务: {file_id}...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }
    response = requests.post(f"{API_BASE_URL}/batches", headers=headers, json=payload)
    response.raise_for_status()
    batch_object = response.json()
    print(f"批量任务创建成功。任务 ID: {batch_object['id']}\n")
    return batch_object

def retrieve_batch(api_key, batch_id):
    """获取批量任务的状态"""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{API_BASE_URL}/batches/{batch_id}", headers=headers)
    response.raise_for_status()
    return response.json()

def download_file_content(api_key, file_id):
    """下载文件的内容"""
    print(f"正在下载结果文件内容，文件 ID: {file_id}...")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(f"{API_BASE_URL}/files/{file_id}/content", headers=headers)
    response.raise_for_status()
    print("内容下载成功。\n")
    return response.text

def process_batch_results(result_content, original_input_file, output_dataset_path):
    """
    处理批量任务的结果，将其转换为微调所需的数据集格式。
    """
    print("开始处理批量任务结果，生成最终的微调数据集...")

    # 1. 从原始输入文件中加载 custom_id 到 instruction 的映射
    print(f"正在读取原始输入文件: {original_input_file}")
    id_to_instruction = {}
    with open(original_input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                request_line = json.loads(line)
                custom_id = request_line.get("custom_id")
                # 提取 "user" 角色的 content 作为 instruction
                user_message = next((msg for msg in request_line["body"]["messages"] if msg["role"] == "user"), None)
                if custom_id and user_message:
                    id_to_instruction[custom_id] = user_message["content"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: 跳过格式错误的行: {line.strip()} - {e}")
                continue
    
    print(f"成功加载 {len(id_to_instruction)} 条原始指令。")

    # 2. 处理结果文件内容
    print("正在解析批量任务返回的结果...")
    formatted_data = []
    lines = result_content.strip().split('\n')
    
    for line in lines:
        try:
            result_line = json.loads(line)
            if result_line.get("error") is not None:
                # 跳过出错的请求
                continue

            custom_id = result_line.get("custom_id")
            instruction = id_to_instruction.get(custom_id)
            
            # 提取模型生成的答案
            response_body = result_line.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            if instruction and choices:
                output = choices[0].get("message", {}).get("content", "")
                if output:
                    formatted_data.append({
                        "instruction": instruction,
                        "input": "", # 根据我们的场景，input 留空
                        "output": output
                    })
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"警告: 解析结果时跳过格式错误的行: {line.strip()} - {e}")
            continue

    # 3. 将格式化后的数据写入最终的训练文件
    print(f"正在将 {len(formatted_data)} 条有效数据写入到: {output_dataset_path}")
    with open(output_dataset_path, 'w', encoding='utf-8') as f_out:
        for item in formatted_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n--- 数据准备完成 ---")
    print(f"成功生成微调数据集，已保存至 {output_dataset_path}")


def main():
    """主函数，按顺序执行所有步骤"""
    parser = argparse.ArgumentParser(description="管理 SiliconFlow 批量任务并处理结果。")
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"用于批量任务的输入 .jsonl 文件。默认为: {DEFAULT_INPUT_FILE}"
    )
    parser.add_argument(
        "--skip_processing",
        action='store_true',
        help="如果只想运行批量任务而不处理结果，请使用此选项。"
    )
    parser.add_argument(
        "--process_only",
        type=str,
        metavar="RESULT_FILE_ID",
        help="仅处理已完成任务的结果文件，需要提供结果文件的 ID。此选项需要联网和API密钥。"
    )
    parser.add_argument(
        "--local_result_file",
        type=str,
        metavar="LOCAL_FILE_PATH",
        help="仅处理本地已下载的结果文件，需要提供文件的路径。此选项无需联网。"
    )

    args = parser.parse_args()

    # 优先处理本地文件，无需 API Key
    if args.local_result_file:
        print(f"正在处理本地结果文件: {args.local_result_file}")
        try:
            with open(args.local_result_file, 'r', encoding='utf-8') as f:
                result_content = f.read()
            # 如果是处理本地文件，我们假设原始输入是那个 "_complete" 的文件
            input_file_to_use = args.input_file
            if input_file_to_use == DEFAULT_INPUT_FILE and os.path.exists(DEFAULT_COMPLETE_INPUT_FILE):
                print(f"检测到完整的输入文件 '{DEFAULT_COMPLETE_INPUT_FILE}'，将使用它来匹配指令。")
                input_file_to_use = DEFAULT_COMPLETE_INPUT_FILE

            process_batch_results(result_content, input_file_to_use, FINAL_TRAIN_DATASET_FILE)
        except FileNotFoundError as e:
            print(f"\n错误: 找不到文件: {e.filename}")
        except Exception as e:
            print(f"\n处理本地文件时出错: {e}")
        return

    # --- 需要 API Key 的流程 ---
    try:
        api_key = get_api_key()

        if args.process_only:
            # 只处理现有结果（从云端下载）
            result_file_id = args.process_only
            result_content = download_file_content(api_key, result_file_id)
            process_batch_results(result_content, args.input_file, FINAL_TRAIN_DATASET_FILE)
            return

        # --- 完整流程 ---
        # 1. 上传文件
        file_object = upload_file(api_key, args.input_file)
        
        # 2. 创建批量任务
        batch_object = create_batch(api_key, file_object['id'])
        batch_id = batch_object['id']

        # 3. 轮询任务状态
        print(f"开始监控任务状态 (ID: {batch_id})。每30秒检查一次...")
        while True:
            batch_status = retrieve_batch(api_key, batch_id)
            status = batch_status.get('status')
            print(f"当前状态: {status}...")

            if status == 'completed':
                print("\n任务完成！")
                if not args.skip_processing:
                    result_file_id = batch_status.get('output_file_id')
                    if result_file_id:
                        # 4. 下载结果
                        result_content = download_file_content(api_key, result_file_id)
                        # 5. 处理结果
                        process_batch_results(result_content, args.input_file, FINAL_TRAIN_DATASET_FILE)
                    else:
                        print("错误: 任务已完成，但未找到输出文件 ID。")
                break
            elif status in ['failed', 'cancelled', 'expired']:
                print(f"\n任务失败或已取消，状态: {status}")
                errors = batch_status.get('errors')
                if errors:
                    print("错误详情:", errors)
                break
            
            time.sleep(30)

    except (ValueError, requests.RequestException) as e:
        print(f"\n操作失败: {e}")

if __name__ == "__main__":
    main()
