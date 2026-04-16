import json
import re
import asyncio
from typing import AsyncGenerator, Dict, Any, Tuple, List
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base import BaseAgent
from prompts.react import react_prompt
from core.llm import config


class ReactAgent(BaseAgent):
    def __init__(
        self,
        mcp_clients: List = None,
        init_model_name: str = "gpt-4o",
        sys_prompt: str = react_prompt,
    ):
        super().__init__(mcp_clients, init_model_name, sys_prompt)

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> AsyncGenerator[str, None]:
        if tool_name == "PARSE_ERROR":
            yield f"Observation: Invalid Action format: {tool_args.get('error')}. Please use valid JSON."
            return

        if tool_name not in self.tool_routes:
            yield f"Observation: Error: Tool {tool_name} not found."
            return

        url, client = self.tool_routes[tool_name]
        try:
            if self.connected:
                result = await client.call_tool(tool_name, tool_args)
            else:
                async with client:
                    result = await client.call_tool(tool_name, tool_args)
            yield f"Observation: {result.data}"
        except Exception as e:
            logging.error(f"【ERROR】Error calling tool {tool_name} from {url}: {e}")
            yield f"Observation: An error occurred while calling tool {tool_name}: {str(e)}"

    async def parse_action(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        # Try strict "Action:" format
        tool_match = re.search(r'(?m)^Action:\s*(\{.*\})', response_text, re.DOTALL)
        if tool_match:
            try:
                tool_data = json.loads(tool_match.group(1).strip())
                return tool_data.get("name"), tool_data.get("arguments")
            except json.JSONDecodeError:
                pass
        
        return None, None

    async def run(self, input: str) -> AsyncGenerator[str, None]:
        if not self.history:
            if not self.tools:
                await self.load_tools()
            self.sys_prompt = self.sys_prompt.format(tools=self.tools)
            self.history.append({"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]})
        
        self.history.append({"role": "user", "content": [{"type": "text", "text": input}]})
        
        async for chunk in self._run_loop():
            yield chunk

    async def _run_loop(self) -> AsyncGenerator[str, None]:
        """Main ReAct reasoning loop, can be reused by subclasses"""
        max_steps = config.get("max_steps", 15)
        step = 0

        while step < max_steps:
            step += 1
            response_text = ""
            
            # Stream generation
            gen = self.llm.async_stream_generate(prompt="", history=self.history)
            
            async for chunk in gen:
                temp_text = response_text + chunk
                obs_index = temp_text.find("Observation:")

                if obs_index != -1:
                    valid_text = temp_text[:obs_index]
                    
                    # Yield only the new valid part
                    chunk_to_yield = valid_text[len(response_text):]
                    if chunk_to_yield:
                        yield chunk_to_yield
                    
                    response_text = valid_text.strip()
                    break
                else:
                    response_text += chunk
                    yield chunk
            
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

            # Parse action from the final (possibly truncated) text
            tool_name, tool_args = await self.parse_action(response_text)
            
            if not tool_name or tool_args is None:
                if "Final Answer" in response_text:
                    return
                
                # Feedback to the model about parsing failure
                logging.warning(f"[ReactAgent] Parsing failed at step {step}. Feedback sent to model.")
                error_msg = "Observation: Failed to parse Action. Please make sure to provide a valid JSON object within 'Action: { ... }' or provide a 'Final Answer'."
                yield "\n" + error_msg
                self.history.append({"role": "user", "content": [{"type": "text", "text": error_msg}]})
                continue

            # Execute Tool
            tool_response_text = ""
            async for tool_chunk in self.call_tool(tool_name, tool_args):
                tool_response_text += tool_chunk
                yield "\n" + tool_chunk

            # Append Observation to history
            self.history.append({"role": "user", "content": [{"type": "text", "text": tool_response_text}]})
        
        logging.warning(f"[ReactAgent] Reached maximum steps ({max_steps}). Stopping task.")
