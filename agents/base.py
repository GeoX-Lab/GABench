import json
import re
import asyncio
from typing import AsyncGenerator, Dict, Any, Tuple, List
import logging
from rich.console import Console
import sys
from contextlib import AsyncExitStack
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.llm import LLM, config
from prompts.base import sys_prompt


class BaseAgent:
    def __init__(
        self,
        mcp_clients: List,
        init_model_name: str = "gpt-4o",
        sys_prompt: str = sys_prompt,
    ):
        self.llm = LLM(init_model_name)

        # Initialize MCP clients
        self.mcp_clients = mcp_clients

        self.tools = []

        # 存储工具路由信息
        self.tool_routes = {}

        self.sys_prompt = sys_prompt

        self.history = []
        self.exit_stack = AsyncExitStack()
        self.connected = False
        
    async def __aenter__(self):
        """Initialize all MCP clients and keep connections open."""
        if self.connected:
            return self
            
        for url, client in self.mcp_clients:
            try:
                await self.exit_stack.enter_async_context(client)
            except Exception as e:
                logging.error(f"【ERROR】Failed to connect to MCP client {url}: {e}")
        
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close all MCP client connections."""
        await self.exit_stack.aclose()
        self.connected = False

    async def load_tools(self):
        # If not persistently connected, use temporary connection for loading tools
        
        for url, client in self.mcp_clients:
            try:
                # Use existing connection if available, otherwise context manager
                if self.connected:
                    tools = await client.list_tools()
                else:
                    async with client:
                        tools = await client.list_tools()
                        
                for tool in tools:
                    if tool.name in self.tool_routes:
                        logging.warning(f"【WARNING】Tool {tool.name} from {url} is duplicated.")
                    self.tool_routes[tool.name] = (url, client)
                self.tools.extend(tools)
            except Exception as e:
                logging.error(f"【ERROR】Cannot load MCP tools from {url}: {e}")
                continue
        
        if not self.tools:
            logging.critical("【CRITICAL】No tools loaded successfully. Exiting program.")
            sys.exit(1)
        
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> AsyncGenerator[str, None]:
        if tool_name not in self.tool_routes:
            yield f"<tool_response>\nError: Tool {tool_name} not found.\n</tool_response>"
            return
        
        url, client = self.tool_routes[tool_name]
        timeout = config.get("tool_timeout", 300)
        try:
            if self.connected:
                 result = await asyncio.wait_for(client.call_tool(tool_name, tool_args), timeout=timeout)
            else:
                async with client:
                    result = await asyncio.wait_for(client.call_tool(tool_name, tool_args), timeout=timeout)
            
            yield f"<tool_response>\n{result.data}\n</tool_response>"
        except asyncio.TimeoutError:
            logging.error(f"【ERROR】Tool {tool_name} execution timed out")
            yield f"<tool_response>\nError: Tool {tool_name} execution timed out.\n</tool_response>"
        except Exception as e:
            logging.error(f"【ERROR】Error calling tool {tool_name} from {url}: {e}")
            yield f"<tool_response>\nAn error occurred while calling tool {tool_name}: {e}\n</tool_response>"


    async def parse_action(self, response: str) -> Tuple[str, Dict[str, Any]]:
        tool_match = re.search(r"<tool_call>\n(.*?)\n</tool_call>", response, re.DOTALL)
        if tool_match:
            tool_call = tool_match.group(1)
            try:
                tool_data = json.loads(tool_call)
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]
                return tool_name, tool_args
            except json.JSONDecodeError:
                logging.error(f"【ERROR】Invalid tool call format: {tool_call}")
                return None, None
        return None, None

    async def run(self, input: str) -> AsyncGenerator[str, None]:
        if not self.history:
            await self.load_tools()
            self.sys_prompt = self.sys_prompt.format(tools=self.tools)
            self.history.append({"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]})
        
        self.history.append({"role": "user", "content": [{"type": "text", "text": input}]})

        # 首次模型回复（可能包含工具调用）
        response_text = ""
        async for chunk in self.llm.async_stream_generate(prompt="", history=self.history):
            response_text += chunk
            yield chunk
            if "</tool_call>" in response_text:
                response_text = response_text[:response_text.find("</tool_call>") + 12]
                break

        # 循环：只要模型输出包含 <tool_call> 就执行工具并继续回复
        max_steps = config.get("max_steps", 30)
        current_step = 0

        while current_step < max_steps:
            tool_name, tool_args = await self.parse_action(response_text)
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
            if not tool_name or tool_args is None:
                # 无工具调用：返回
                return
            
            current_step += 1

            # 执行工具并流式输出工具结果
            tool_response_text = ""
            async for tool_chunk in self.call_tool(tool_name, tool_args):
                tool_response_text += tool_chunk
                yield tool_chunk

            # 将工具结果加入对话历史
            self.history.append({"role": "user", "content": [{"type": "text", "text": tool_response_text}]})

            # 使用工具结果让模型继续生成回答（可能再次触发 <tool_call>）
            response_text = ""
            async for chunk in self.llm.async_stream_generate(prompt="", history=self.history):
                response_text += chunk
                yield chunk
                if "</tool_call>" in response_text:
                    response_text = response_text[:response_text.find("</tool_call>") + 12]
                    break
        
        logging.warning(f"[BaseAgent] Reached maximum steps ({max_steps}). Stopping task.")
    
    async def print_history(self):
        console = Console()
        color_map = {
            "SYSTEM": "yellow",
            "USER": "cyan",
            "ASSISTANT": "green",
        }
        console.print("\n")

        for item in self.history:
            role = str(item.get("role", "unknown")).upper()
            color = color_map.get(role, "white")

            console.print(f"[white]{'-' * 80}[/]")

            console.print(f"[bold {color}]ROLE: {role}[/]")

            contents = item.get("content") or []
            if not contents:
                console.print(f"[{color}]    (no content)[/]")
                continue

            for content in contents:
                ctype = content.get("type", "text")
                if ctype == "text":
                    ctext = content.get("text", "")
                    console.print(f"[{color}]{ctext}[/]")
                else:
                    console.print(f"[{color}]{str(content)}[/]")

    async def clear_history(self):
        self.history = []
        self.history.append({"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]})