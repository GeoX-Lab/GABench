from typing import AsyncGenerator, Dict, Any, Tuple, List
import re
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base import BaseAgent
from agents.react import ReactAgent
from prompts.plan_and_react import planner_prompt, reactor_prompt


class PlanAgent(BaseAgent):
    def __init__(
        self,
        mcp_clients: List = None,
        init_model_name: str = "gpt-4o",
        sys_prompt: str = planner_prompt,
    ):
        super().__init__(mcp_clients, init_model_name, sys_prompt)
        self.subtasks = ""  # 使用实例变量而非全局变量

    async def run(self, input: str) -> AsyncGenerator[str, None]:
        if not self.history:
            await self.load_tools()
            
            # Only provide tool names and descriptions for the planner to save context
            tools_desc = []
            for tool in self.tools:
                # Handle both object and dict access for compatibility
                name = getattr(tool, 'name', None) or tool.get('name')
                desc = getattr(tool, 'description', None) or tool.get('description', '')
                
                # Truncate description to remove detailed Args/Returns to save tokens
                if desc:
                    desc = desc.split('Args:')[0].split('Returns:')[0].strip()
                
                tools_desc.append(f"- {name}: {desc}")
            
            self.sys_prompt = self.sys_prompt.format(tools="\n".join(tools_desc))
            self.history.append({"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]})

        self.history.append({"role": "user", "content": [{"type": "text", "text": input}]})

        response_text = ""
        async for chunk in self.llm.async_stream_generate(prompt="", history=self.history):
            response_text += chunk
            yield chunk

        self.history.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        
        # Extract JSON part for solver
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if json_match:
            self.subtasks = json_match.group(1)
        else:
            # Fallback: try to find list brackets directly if markdown is missing
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                self.subtasks = json_match.group(1)


class SolveReactAgent(ReactAgent):
    def __init__(
        self,
        mcp_clients: List = None,
        init_model_name: str = "gpt-4o",
        sys_prompt: str = reactor_prompt,
    ):
        super().__init__(mcp_clients, init_model_name, sys_prompt)
        self.subtasks = ""

    def set_subtasks(self, subtasks: str):
        """Set subtasks from PlanAgent"""
        self.subtasks = subtasks

    async def run(self, input: str) -> AsyncGenerator[str, None]:
        # Override only to inject subtasks into system prompt
        if not self.history:
            if not self.tools:
                await self.load_tools()
            self.sys_prompt = self.sys_prompt.format(tools=self.tools, subtasks=self.subtasks)
            self.history.append({"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]})
        
        # Use parent's run logic by delegating to it
        self.history.append({"role": "user", "content": [{"type": "text", "text": input}]})
        
        # Call parent's loop logic
        async for chunk in super()._run_loop():
            yield chunk
