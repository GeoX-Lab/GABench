import json
import re
import logging
import sys
from typing import AsyncGenerator, List, Dict, Any
from pathlib import Path

# Add parent directory to path to allow imports from core/prompts
sys.path.append(str(Path(__file__).parent.parent))

from agents.base import BaseAgent
from prompts.plan_and_solve import planner_prompt #, summary_prompt

class PlanAndSolveAgent(BaseAgent):
    def __init__(
        self,
        mcp_clients: List = None,
        init_model_name: str = "gpt-4o",
        sys_prompt: str = planner_prompt,
    ):
        super().__init__(mcp_clients, init_model_name, sys_prompt)
        # self.summary_prompt = summary_prompt

    async def run(self, input: str) -> AsyncGenerator[str, None]:
        # 1. Initialize tools if not loaded
        if not self.tools:
            await self.load_tools()
        
        # Prepare tool descriptions for the prompt
        tools_info = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ])

        # --- Step 1: Planning ---
        # yield "Thinking Process:\n1. Generating Plan...\n"
        
        planning_prompt_formatted = self.sys_prompt.format(
            tools=tools_info,
            query=input
        )
        
        # Use a fresh history for planning
        messages = [
            {"role": "system", "content": [{"type": "text", "text": planning_prompt_formatted}]},
            {"role": "user", "content": [{"type": "text", "text": input}]}
        ]
        
        # Record to history for logging
        self.history.append(messages[0])
        self.history.append(messages[1])

        plan_response_text = ""
        async for chunk in self.llm.async_stream_generate(prompt="", history=messages):
            plan_response_text += chunk
            yield chunk 
        
        yield "\n"
        
        # Record plan to history
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": plan_response_text}]})

        # Parse Plan
        plan_steps = []
        try:
            # Try to find JSON block
            json_match = re.search(r"```json\s*(.*?)\s*```", plan_response_text, re.DOTALL)
            if json_match:
                plan_json_str = json_match.group(1)
            else:
                # Fallback: Try to find the first JSON object or list
                json_match = re.search(r"(\{.*\}|$$.*$$)", plan_response_text, re.DOTALL)
                if json_match:
                    plan_json_str = json_match.group(1)
                else:
                    # Attempt to parse the whole text if it looks like json
                    plan_json_str = plan_response_text

            # Clean up JSON string: remove comments // and /* */
            plan_json_str = re.sub(r'//.*', '', plan_json_str)
            plan_json_str = re.sub(r'/\*.*?\*/', '', plan_json_str, flags=re.DOTALL)
            
            plan_data = json.loads(plan_json_str)
            
            # Handle different JSON structures
            if isinstance(plan_data, dict) and "plan" in plan_data:
                plan_steps = plan_data["plan"]
            elif isinstance(plan_data, list):
                plan_steps = plan_data
            elif isinstance(plan_data, dict):
                 plan_steps = [plan_data]

        except Exception as e:
            logging.error(f"Plan parsing failed: {e}")
            yield f"\n[ERROR] Failed to parse plan: {e}\n"
            return

        if not plan_steps:
            yield "\n[ERROR] Empty plan generated.\n"
            return

        # --- Step 2: Execution ---
        # yield "\nThinking Process:\n2. Executing Plan...\n"
        
        execution_results = []
        
        for i, step in enumerate(plan_steps):
            tool_name = step.get("tool")
            parameters = step.get("parameters", {})
            
            # yield f"\n[Step {i+1}] Executing {tool_name}...\n"
            
            step_result = ""
            try:
                # Record tool execution attempt
                self.history.append({
                    "role": "assistant", 
                    "content": [{"type": "text", "text": f"Executing Step {i+1}: {tool_name}({json.dumps(parameters)})"}]
                })

                # Use BaseAgent.call_tool
                async for chunk in self.call_tool(tool_name, parameters):
                    step_result += chunk
                    yield chunk
                
                # Strip XML tags if present
                clean_result = re.sub(r"</?tool_response>", "", step_result).strip()
                execution_results.append(f"Step {i+1} result: {clean_result}")
                
                # Record tool result
                self.history.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": f"Step {i+1} Result: {clean_result}"}]
                })
                
            except Exception as e:
                error_msg = f"Tool execution error: {str(e)}"
                execution_results.append(f"Step {i+1} result: {error_msg}")
                yield f"\n{error_msg}\n"
                
                # Record error
                self.history.append({
                    "role": "user", 
                    "content": [{"type": "text", "text": f"Step {i+1} Error: {error_msg}"}]
                })

        # --- Step 3: Summary (Commented Out) ---
        # yield "\n\nThinking Process:\n3. Summarizing Results...\n"
        
        # results_text = "\n".join(execution_results)
        # plan_json_str = json.dumps(plan_steps, ensure_ascii=False, indent=2)
        
        # summary_prompt_formatted = self.summary_prompt.format(
        #     tools=tools_info,
        #     query=input,
        #     plan=plan_json_str,
        #     results=results_text
        # )
        
        # formatted_summary_messages = [
        #      {"role": "user", "content": [{"type": "text", "text": summary_prompt_formatted}]}
        # ]
        
        # async for chunk in self.llm.async_stream_generate(prompt="", history=formatted_summary_messages):
        #     yield chunk