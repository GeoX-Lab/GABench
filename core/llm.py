import os
import yaml
import httpx
import base64
import aiofiles
import traceback
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import AsyncGenerator, Union


load_dotenv()

with open("config.yaml", "r") as f:
    raw_config = os.path.expandvars(f.read())
    config = yaml.safe_load(raw_config)
LLM_CONFIG = config["llm"]
LOG_PATH = config["log_path"]
MAX_TOKENS = config.get("max_tokens", 4096)


class LLMUsageLogger:
    """Generic logger for LLM usage metrics."""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, log_type: str, data: dict):
        """Log without task context injection."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": log_type,
            **data
        }
        
        max_retries = 3
        for i in range(max_retries):
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                break
            except Exception as e:
                if i == max_retries - 1:
                    print(f"Logging failed after {max_retries} attempts: {e}")
                else:
                    time.sleep(0.1)


class LLM:
    def __init__(self, model: str="gpt-4o"):
        cfg = LLM_CONFIG.get(model)
        if cfg is None:
            raise ValueError(f"Model '{model}' not found in config.yaml")
        self.async_client = AsyncOpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            http_client=httpx.AsyncClient(verify=False, timeout=300)
        )
        self.model = cfg["model"]
        self.logger = LLMUsageLogger(LOG_PATH)


    async def async_generate(
        self,
        prompt: str,
        image_path: Union[str, Path, None] = None,
        history: list[dict] = None
    ) -> str:
        start_time = time.time()
        try:
            messages = await self.prepare_messages(prompt, image_path, history)

            chat_response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=MAX_TOKENS
            )
            
            end_time = time.time()
            content = chat_response.choices[0].message.content
            usage = chat_response.usage
            
            # Calculate metrics
            total_duration = end_time - start_time
            completion_tokens = usage.completion_tokens if usage else 0
            
            metrics = {
                "ttft": total_duration, # For non-stream, TTFT is total time
                "total_time": total_duration,
                "speed": completion_tokens / total_duration if total_duration > 0 else 0,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": completion_tokens,
                "total_tokens": usage.total_tokens if usage else 0
            }
            
            self.logger.log("llm_generation", {
                "model": self.model,
                "metrics": metrics
            })
            
            return content

        except Exception as e:
            self.handle_error(e)
            raise e

    async def async_stream_generate(
        self,
        prompt: str,
        image_path: Union[str, Path, None] = None,
        history: list[dict] = None
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()
        first_token_time = None
        collected_content = []
        usage_info = None
        
        try:
            messages = await self.prepare_messages(prompt, image_path, history)

            async for chunk in await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=MAX_TOKENS
            ):
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage

                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        content = choice.delta.content
                        if content is not None:
                            if first_token_time is None:
                                first_token_time = time.time()
                            collected_content.append(content)
                            yield content
            
            # Log after stream finishes
            end_time = time.time()
            ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
            
            prompt_tokens = usage_info.prompt_tokens if usage_info else 0
            completion_tokens = usage_info.completion_tokens if usage_info else 0
            total_tokens = usage_info.total_tokens if usage_info else 0
            
            generation_time = end_time - (first_token_time or start_time)
            speed = completion_tokens / generation_time if generation_time > 0 else 0
            
            metrics = {
                "ttft": ttft,
                "total_time": end_time - start_time,
                "speed": speed,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            self.logger.log("llm_generation", {
                "model": self.model,
                "metrics": metrics
            })

        except Exception as e:
            self.handle_error(e)
            raise e

    async def prepare_messages(
        self,
        prompt: str,
        image_path: Union[str, Path, None],
        history: list[dict] = None
    ) -> list[dict]:
        messages = history.copy() if history else []

        if image_path:
            base64_image = await self.image_to_base64(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
            messages.append(
                {"role": "user", "content": content}
            )
        elif prompt:
            content = [{"type": "text", "text": prompt}]
            messages.append(
                {"role": "user", "content": content}
            )
        
        return messages

    def handle_error(self, e: Exception) -> str:
        print(f"==========Error: {e}==========")
        print(traceback.format_exc())
        print(f"==========Model: {self.model}==========")
        return f"ERROR: {type(e).__name__} - {str(e)}"

    async def image_to_base64(self, image_path: Union[str, Path]) -> str:
        async with aiofiles.open(image_path, "rb") as image_file:
            content = await image_file.read()
            encoded_string = base64.b64encode(content).decode("utf-8")
        return encoded_string

if __name__ == "__main__":
    print(LLM_CONFIG)

    import asyncio


    async def test():
        llm = LLM("gpt-4o")

        history = [
            {"role": "user", "content": [{"type": "text", "text": "你是三年一班的龙傲天"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "是的，我是三年一班的龙傲天"}]}
        ]

        result = await llm.async_generate("你好，介绍一下你自己")
        print(result)

        async for chunk in llm.async_stream_generate("你好，介绍一下你自己", history=history):
            print(chunk, end="")


    asyncio.run(test())