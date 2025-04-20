"""
增量生成模块 - 实现"边想边说"的核心功能
"""

import os
import re
import torch
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from queue import Queue
from threading import Thread
from dataclasses import dataclass

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IncrementalGeneration:
    """
    增量生成类，实现"边想边说"的功能
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    def __post_init__(self):
        # 初始化流式生成器
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 确保模型在正确的设备上
        self.device = self.model.device
        
        # 思考标记的正则表达式
        self.think_pattern = re.compile(
            f"{re.escape(config.THINK_START_TOKEN)}(.*?){re.escape(config.THINK_END_TOKEN)}",
            re.DOTALL
        )
        
        # 代码运行标记的正则表达式
        self.code_run_pattern = re.compile(r"```(\w+)\s(.*?)```\s*RUN", re.DOTALL)
        
    def format_prompt(self, user_input: str) -> str:
        """
        格式化用户输入为模型输入
        """
        return f"用户: {user_input}\n助手: "
    
    def generate_incremental(self, 
                           user_input: str, 
                           callback: Optional[Callable[[str, bool], Any]] = None,
                           interrupt_event: Optional[threading.Event] = None,
                           additional_input: Optional[str] = None) -> str:
        """
        增量生成响应
        
        Args:
            user_input: 用户输入
            callback: 回调函数，接收生成的文本片段和是否为思考内容的标志
            interrupt_event: 中断事件，用于实现中断生成
            additional_input: 中断后的额外输入
            
        Returns:
            生成的完整响应
        """
        # 准备输入
        if additional_input:
            # 处理中断后的情况
            prompt = f"{user_input}\n用户: {additional_input}\n助手: "
        else:
            prompt = self.format_prompt(user_input)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 创建生成线程
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "streamer": self.streamer
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 收集完整输出
        full_output = ""
        current_segment = ""
        is_in_thinking = False
        is_in_code = False
        code_buffer = ""
        code_lang = ""
        
        # 处理流式输出
        for new_text in self.streamer:
            if interrupt_event and interrupt_event.is_set():
                logger.info("生成被中断")
                break
            
            full_output += new_text
            
            # 更新当前段落
            current_segment += new_text
            
            # 检查是否进入或退出思考模式
            if config.THINK_START_TOKEN in new_text and not is_in_thinking:
                is_in_thinking = True
                current_segment = new_text.split(config.THINK_START_TOKEN, 1)[1]
            elif config.THINK_END_TOKEN in new_text and is_in_thinking:
                is_in_thinking = False
                thought_content = current_segment.split(config.THINK_END_TOKEN, 1)[0]
                current_segment = current_segment.split(config.THINK_END_TOKEN, 1)[1] if len(current_segment.split(config.THINK_END_TOKEN, 1)) > 1 else ""
                
                # 回调，通知有新的思考内容
                if callback:
                    callback(thought_content, True)
            
            # 检查是否进入或退出代码模式
            if "```" in new_text:
                if not is_in_code:
                    # 可能进入代码模式
                    parts = new_text.split("```", 1)
                    if len(parts) > 1 and len(parts[1].strip()) > 0:
                        is_in_code = True
                        lang_and_code = parts[1].strip()
                        if " " in lang_and_code:
                            code_lang, initial_code = lang_and_code.split(" ", 1)
                            code_buffer = initial_code
                        else:
                            code_lang = lang_and_code
                            code_buffer = ""
                else:
                    # 可能退出代码模式
                    if "```" in new_text:
                        is_in_code = False
                        # 检查是否带有 RUN 标记
                        if "RUN" in new_text.split("```")[1]:
                            # 回调，通知有可执行代码
                            if callback:
                                callback(f"```{code_lang}\n{code_buffer}```RUN", False)
                            code_buffer = ""
                            code_lang = ""
                        else:
                            # 普通代码块
                            if callback:
                                callback(f"```{code_lang}\n{code_buffer}```", False)
                            code_buffer = ""
                            code_lang = ""
            elif is_in_code:
                # 在代码模式中，继续收集代码
                code_buffer += new_text
            elif not is_in_thinking and not is_in_code and new_text.strip() and callback:
                # 普通文本，非思考，非代码
                callback(new_text, False)
        
        thread.join()
        return full_output
    
    async def generate_incremental_async(self, 
                                       user_input: str, 
                                       callback: Optional[Callable[[str, bool], Any]] = None,
                                       interrupt_event: Optional[threading.Event] = None,
                                       additional_input: Optional[str] = None) -> str:
        """
        异步版本的增量生成响应
        
        Args:
            user_input: 用户输入
            callback: 回调函数，接收生成的文本片段和是否为思考内容的标志
            interrupt_event: 中断事件，用于实现中断生成
            additional_input: 中断后的额外输入
            
        Returns:
            生成的完整响应
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_incremental(
                user_input=user_input,
                callback=callback,
                interrupt_event=interrupt_event,
                additional_input=additional_input
            )
        )
    
    def extract_thoughts(self, text: str) -> List[str]:
        """
        从生成的文本中提取思考部分
        """
        return [match.group(1) for match in self.think_pattern.finditer(text)]
    
    def extract_executable_code(self, text: str) -> List[Dict[str, str]]:
        """
        从生成的文本中提取可执行代码
        """
        results = []
        for match in self.code_run_pattern.finditer(text):
            lang = match.group(1)
            code = match.group(2)
            results.append({
                "language": lang,
                "code": code
            })
        return results