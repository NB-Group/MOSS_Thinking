"""
推理脚本 - 演示"边想边说"的效果
"""

import os
import time
import threading
import argparse
import logging
import sys
import re
from typing import List, Dict, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from utils.incremental_generation import IncrementalGeneration
from utils.model_utils import load_tokenizer_and_model
from modelscope import snapshot_download
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class IncrementalChatDemo:
    """
    增量式聊天演示
    """
    def __init__(self, model_path: str):
        """
        初始化聊天环境
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        logger.info(f"从 {model_path} 加载模型")
        self.tokenizer, self.model = load_tokenizer_and_model(model_path, use_lora=False)
        
        # 初始化增量生成器
        self.incremental_generator = IncrementalGeneration(
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        # 用于中断生成的事件
        self.interrupt_event = threading.Event()
        
        # 保存对话历史
        self.conversation_history = []
        
        # 颜色定义
        self.THINK_COLOR = "\033[33m"  # 黄色
        self.USER_COLOR = "\033[36m"   # 青色
        self.ASSISTANT_COLOR = "\033[32m"  # 绿色
        self.CODE_COLOR = "\033[35m"   # 紫色
        self.RESET_COLOR = "\033[0m"   # 重置颜色
        
    def print_colored(self, text: str, color: str):
        """打印带颜色的文本"""
        print(f"{color}{text}{self.RESET_COLOR}", end="", flush=True)
    
    def handle_output(self, text: str, is_thought: bool):
        """
        处理模型输出
        
        Args:
            text: 生成的文本
            is_thought: 是否是思考内容
        """
        if is_thought:
            self.print_colored(text, self.THINK_COLOR)
        elif text.startswith("```") and text.endswith("```RUN"):
            # 处理可执行代码
            self.print_colored(text, self.CODE_COLOR)
        elif text.startswith("```") and text.endswith("```"):
            # 处理普通代码
            self.print_colored(text, self.CODE_COLOR)
        else:
            # 普通文本
            self.print_colored(text, self.ASSISTANT_COLOR)
    
    def run_conversation(self):
        """
        运行对话演示
        """
        try:
            print("欢迎使用增量式思考演示系统!")
            print("输入 'q' 或 'quit' 退出, 输入 'c' 或 'ctrl+c' 中断当前生成")
            
            while True:
                # 获取用户输入
                self.print_colored("\n用户: ", self.USER_COLOR)
                user_input = input()
                
                if user_input.lower() in ["q", "quit", "exit"]:
                    print("感谢使用，再见！")
                    break
                
                # 重置中断事件
                self.interrupt_event.clear()
                
                # 启动一个线程监听中断命令
                def check_interrupt():
                    while not self.interrupt_event.is_set():
                        if input() in ["c", "ctrl+c"]:
                            print("\n[生成已中断]")
                            self.interrupt_event.set()
                            return
                        time.sleep(0.1)
                
                interrupt_thread = threading.Thread(target=check_interrupt)
                interrupt_thread.daemon = True
                interrupt_thread.start()
                
                # 生成响应
                self.print_colored("助手: ", self.ASSISTANT_COLOR)
                
                # 如果被中断，获取额外输入
                additional_input = None
                if self.interrupt_event.is_set():
                    self.print_colored("\n用户补充: ", self.USER_COLOR)
                    additional_input = input()
                
                # 使用增量生成器生成响应
                full_response = self.incremental_generator.generate_incremental(
                    user_input=user_input,
                    callback=self.handle_output,
                    interrupt_event=self.interrupt_event,
                    additional_input=additional_input
                )
                
                # 保存对话历史
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": full_response
                })
                
        except KeyboardInterrupt:
            print("\n程序已终止")
        except Exception as e:
            logger.error(f"发生错误: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="运行增量式聊天演示")
    parser.add_argument("--model_path", type=str, default=None,
                      help="模型路径，如果不指定则使用配置中的模型")
    parser.add_argument("--model_id", type=str, default=config.STUDENT_MODEL_ID,
                      help=f"从ModelScope加载的模型ID，默认为{config.STUDENT_MODEL_ID}")
    parser.add_argument("--local_model", action="store_true",
                      help="使用本地保存的模型，默认为False")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                      help=f"本地模型目录，默认为{config.OUTPUT_DIR}")
    
    args = parser.parse_args()
    
    # 确定模型路径
    model_path = args.model_path
    if model_path is None:
        if args.local_model:
            model_path = args.output_dir
            if not os.path.exists(model_path):
                raise ValueError(f"本地模型路径不存在: {model_path}")
        else:
            logger.info(f"从ModelScope下载模型: {args.model_id}")
            model_path = snapshot_download(args.model_id, cache_dir=config.CACHE_DIR)
    
    # 创建演示实例
    demo = IncrementalChatDemo(model_path)
    
    # 运行对话
    demo.run_conversation()

if __name__ == "__main__":
    main()