import json
import argparse
import asyncio
import re
import traceback
import sys
import logging
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Dict, Any, Tuple

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from core.mcp_client import get_mcp_clients

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ReplayLogs")

async def load_tools(mcp_clients_list: list) -> Dict[str, Any]:
    """Load available tools from all MCP clients."""
    tool_map = {}
    for _, client in mcp_clients_list:
        try:
            # client is already entered in main context
            tools = await client.list_tools()
            for tool in tools:
                tool_map[tool.name] = client
        except Exception as e:
            logger.error(f"Error listing tools for client: {e}")
    return tool_map

async def execute_action(tool_name: str, tool_args: dict, tool_map: Dict[str, Any]):
    """Execute a single tool action."""
    if tool_name not in tool_map:
        logger.warning(f"  [SKIP] Tool '{tool_name}' not found in available tools.")
        return

    client = tool_map[tool_name]
    try:
        logger.info(f"  [EXEC] {tool_name}...")
        # We await the result to ensure sequential execution
        # We assume the environment (like GEOBENCH_OUTPUT_DIR) is handled by the tool implementation
        # or defaults are used.
        result = await client.call_tool(tool_name, tool_args)
        logger.info(f"  [OBSERVATION] {str(result.data)[:200]}...")
    except Exception as e:
        logger.error(f"  [ERR] Failed to execute {tool_name}: {e}")
        logger.error(f"  [ARGS] {tool_args}")
        logger.error(traceback.format_exc())

async def replay_task(task_data: dict, tool_map: Dict[str, Any]):
    """Replay all actions in a task's history."""
    task_id = task_data.get('task_id')
    history = task_data.get('history', [])
    
    logger.info(f"--- Replaying Task {task_id} ---")
    
    action_count = 0
    
    for msg in history:
        role = msg.get('role')
        if role == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, list):
                 content = "".join([p.get('text', '') for p in content if p.get('type') == 'text'])
            
            # Match Action block similar to ReactAgent
            # We look for "Action: { ... }"
            # Using DOTALL to match multiline JSON
            action_match = re.search(r'(?m)^Action:\s*(\{.*\})', content, re.DOTALL)
            
            if action_match:
                try:
                    action_json = action_match.group(1).strip()
                    action_data = json.loads(action_json)
                    
                    tool_name = action_data.get("name")
                    tool_args = action_data.get("arguments", {})
                    
                    if tool_name:
                        await execute_action(tool_name, tool_args, tool_map)
                        action_count += 1
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error parsing action: {e}")

    logger.info(f"--- Task {task_id} Done. Executed {action_count} actions. ---")

async def main():
    parser = argparse.ArgumentParser(description="Replay Tool Logic from LLM Logs (No LLM Inference)")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the .jsonl log file")
    parser.add_argument("--task_id", type=str, help="Specific task ID to replay (optional)")
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        return

    logger.info("Initializing MCP Clients...")
    mcp_clients = get_mcp_clients()
    
    async with AsyncExitStack() as stack:
        # Connect all clients
        logger.info("Connecting to tools...")
        for _, client in mcp_clients:
            await stack.enter_async_context(client)
        
        # Load tools map
        tool_map = await load_tools(mcp_clients)
        logger.info(f"Loaded {len(tool_map)} tools.")

        # Load tasks
        tasks = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    # If specific task requested, filter here
                    if args.task_id and str(data.get('task_id')) != args.task_id:
                        continue
                    tasks.append(data)
                except:
                    continue
        
        logger.info(f"Found {len(tasks)} tasks to replay.")
        
        for task in tasks:
            await replay_task(task, tool_map)

if __name__ == "__main__":
    asyncio.run(main())