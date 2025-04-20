"""
生产环境API服务 - 提供"边想边说"功能的HTTP API
"""

import os
import time
import argparse
import logging
import asyncio
import threading
import uvicorn
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modelscope import snapshot_download

from utils.model_utils import load_tokenizer_and_model
from utils.incremental_generation import IncrementalGeneration
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("service.log")
    ]
)
logger = logging.getLogger(__name__)

# 定义请求模型
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    additional_input: Optional[str] = None

class IncrementalServiceManager:
    """增量生成服务管理器"""
    
    def __init__(self, model_path: str):
        """
        初始化服务管理器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.is_ready = False
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            logger.info(f"从 {self.model_path} 加载模型")
            self.tokenizer, self.model = load_tokenizer_and_model(self.model_path, use_lora=False)
            
            # 初始化生成器
            self.generator = IncrementalGeneration(
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            self.is_ready = True
            logger.info("模型加载完成，服务就绪")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    async def generate_stream(self, 
                           prompt: str, 
                           websocket: WebSocket,
                           max_new_tokens: int = 1024,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           repetition_penalty: float = 1.1,
                           additional_input: Optional[str] = None):
        """
        流式生成响应并通过WebSocket发送
        
        Args:
            prompt: 用户输入
            websocket: WebSocket连接
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_p: Top-p采样参数
            repetition_penalty: 重复惩罚参数
            additional_input: 中断后的额外输入
        """
        if not self.is_ready:
            await websocket.send_json({"error": "模型尚未加载完成"})
            return
        
        # 设置生成参数
        self.generator.max_new_tokens = max_new_tokens
        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.repetition_penalty = repetition_penalty
        
        # 中断事件
        interrupt_event = threading.Event()
        
        # 回调函数，处理生成的文本片段
        async def callback(text: str, is_thought: bool):
            try:
                await websocket.send_json({
                    "text": text,
                    "is_thought": is_thought,
                    "is_code_run": text.endswith("```RUN") if text.startswith("```") else False,
                    "finished": False
                })
            except Exception as e:
                logger.error(f"WebSocket发送失败: {e}")
                interrupt_event.set()
        
        try:
            # 异步生成响应
            full_response = await self.generator.generate_incremental_async(
                user_input=prompt,
                callback=callback,
                interrupt_event=interrupt_event,
                additional_input=additional_input
            )
            
            # 发送完成信号
            await websocket.send_json({
                "text": "",
                "is_thought": False,
                "finished": True,
                "full_response": full_response
            })
            
        except Exception as e:
            logger.error(f"生成过程出错: {e}")
            await websocket.send_json({"error": f"生成过程出错: {str(e)}"})

# 创建FastAPI应用
app = FastAPI(title="Qwen边想边说 API服务", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务管理器实例
service_manager = None

@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    global service_manager
    # 注意: 服务管理器会在应用启动时创建，解析参数在main函数中完成

@app.get("/")
async def root():
    """API根路径"""
    return {
        "status": "ok",
        "model": os.path.basename(service_manager.model_path) if service_manager else "未加载",
        "service": "Qwen边想边说 API服务",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "model_loaded": service_manager.is_ready if service_manager else False,
        "timestamp": time.time()
    }

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket端点，用于流式生成响应"""
    await websocket.accept()
    
    try:
        # 等待客户端消息
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # 解析请求参数
            prompt = request_data.get("prompt", "")
            max_new_tokens = request_data.get("max_new_tokens", 1024)
            temperature = request_data.get("temperature", 0.7)
            top_p = request_data.get("top_p", 0.9)
            repetition_penalty = request_data.get("repetition_penalty", 1.1)
            additional_input = request_data.get("additional_input", None)
            
            # 流式生成响应
            await service_manager.generate_stream(
                prompt=prompt,
                websocket=websocket,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                additional_input=additional_input
            )
            
            # 如果客户端发送了"close"，则关闭连接
            if prompt.lower() == "close":
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket连接已关闭")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

def main():
    """主函数"""
    global service_manager
    
    parser = argparse.ArgumentParser(description="运行边想边说API服务")
    parser.add_argument("--model_path", type=str, default="./output",
                      help="模型路径，默认为./output")
    parser.add_argument("--port", type=int, default=8000,
                      help="服务端口，默认为8000")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="服务主机地址，默认为0.0.0.0")
    parser.add_argument("--model_id", type=str, default=None,
                      help="从ModelScope加载的模型ID，如不指定则使用本地模型")
    parser.add_argument("--workers", type=int, default=1,
                      help="工作进程数，默认为1")
    parser.add_argument("--log_level", type=str, default="info",
                      help="日志级别，默认为info")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # 确定模型路径
    model_path = args.model_path
    if args.model_id:
        logger.info(f"从ModelScope下载模型: {args.model_id}")
        model_path = snapshot_download(args.model_id, cache_dir=config.CACHE_DIR)
    
    # 创建服务管理器
    try:
        service_manager = IncrementalServiceManager(model_path)
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        return
    
    # 启动服务
    logger.info(f"启动API服务，地址: {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)

if __name__ == "__main__":
    main()