import argparse
import pandas as pd
import json
import csv
import re
from pathlib import Path
import sys

def load_benchmark_steps(benchmark_path):
    """Load GT Toolchain Length from benchmark CSV."""
    steps_map = {}
    path = Path(benchmark_path)
    if not path.exists():
        print(f"Warning: Benchmark file not found at {path}. Efficiency calculation will be skipped.")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = str(row['ID'])
                try:
                    # 'Toolchain Length' column assumed to exist per step_by_step.py
                    if 'Toolchain Length' in row and row['Toolchain Length']:
                        steps_map[tid] = int(row['Toolchain Length'])
                except:
                    pass
    except Exception as e:
        print(f"Error loading benchmark: {e}")
    return steps_map

def count_actual_steps(history, agent_mode="unknown"):
    """
    Count actual tool calls in history based on agent mode.
    Modes:
    - 'base': Look for <tool_call>...</tool_call>
    - 'react' or 'plan_and_react': Look for Action: {...}
    - 'plan_and_solve': Look for Executing Step X: ... patterns in assistant messages
    """
    count = 0
    if not isinstance(history, list):
        return 0
    
    # Pre-compile regex patterns
    xml_pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    react_pattern = re.compile(r'(?m)^Action:\s*(\{.*\})')
    # For plan_and_solve, we look for our logging marker "Executing Step X: tool_name({...})"
    # or just raw tool calls if BaseAgent.call_tool format is used directly in history
    
    for msg in history:
        role = msg.get('role')
        content = msg.get('content', '')
        
        # Normalize content to string
        if isinstance(content, list):
            text_content = ""
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_content += part.get('text', '')
                elif isinstance(part, str):
                    text_content += part.get('text', '')
            content = text_content
            
        if not isinstance(content, str):
            continue

        if role == 'assistant':
            # 1. Check for OpenAI tool_calls (Universal fallback)
            if 'tool_calls' in msg and msg['tool_calls']:
                count += len(msg['tool_calls'])
                continue
            
            # 2. Mode-specific counting
            if agent_mode in ['base']:
                matches = xml_pattern.findall(content)
                count += len(matches)
                
            elif agent_mode in ['react', 'plan_and_react']:
                matches = react_pattern.findall(content)
                count += len(matches)
                
            elif agent_mode == 'plan_and_solve':
                # In PlanAndSolveAgent, we log: "Executing Step {i+1}: {tool_name}({parameters})"
                # This is a robust way to count executed steps
                if "Executing Step" in content and "Result:" not in content:
                     # Simple check: count occurrences of "Executing Step" lines
                     # But typically one message = one step execution log? 
                     # Let's count lines starting with "Executing Step"
                     matches = re.findall(r'Executing Step \d+:', content)
                     count += len(matches)

    return count

def check_error_encountered(history):
    """
    Detect if the agent encountered an error during execution.
    Criteria: Check for common error keywords in the history (Tools output).
    """
    for i, msg in enumerate(history):
        role = msg.get('role')
        content = msg.get('content', '')
        
        # Skip the initial user prompt (usually index 0)
        if i == 0 and role == 'user':
            continue
            
        # Only check 'user' messages (Observations/Tool Outputs)
        # Skip 'assistant' messages (Thoughts)
        if role != 'user':
            continue

        # Extract text content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
            
        content_str = str(content).strip()
        
        # Only check messages that are explicitly marked as Observations
        if "Observation:" not in content_str:
            continue
            
        # Remove "Observation:" prefix if present to get the raw tool output
        content_str = content_str.split("Observation:", 1)[1].strip()
        
        # STRICT RULE: Valid tool output MUST be a JSON object/array.
        # Any non-JSON output (e.g., plain text error messages, stack traces) is considered an ERROR.
        try:
            # Basic check: must start with { or [ to be a valid structured response
            if not (content_str.startswith('{') or content_str.startswith('[')):
                return True # Not a structured JSON response -> Error
            
            # Full parsing check
            json.loads(content_str)
            
            # If successful, it's a valid tool output
            continue
            
        except Exception:
            # Parsing failed -> It is an error (Traceback, plain text error, etc.)
            return True
                
    return False


def print_metrics(title, stats, total_count=None):
    if total_count is None:
        total_count = stats["total"]
        
    avg_step_efficiency_macro = 0.0
    if stats["success_efficiencies"]:
        avg_step_efficiency_macro = sum(stats["success_efficiencies"]) / len(stats["success_efficiencies"])
        
    avg_step_efficiency_micro = 0.0
    if stats["success_actual_steps_total"] > 0:
        avg_step_efficiency_micro = stats["success_gt_steps_total"] / stats["success_actual_steps_total"]

    print(f"\n--- {title} (Count: {total_count}) ---")
    
    if stats["success_count"] > 0:
        print(f"Eff_macro (Macro-Average Step Efficiency): {avg_step_efficiency_macro:.4f}")
        print(f"Eff_micro (Micro-Average Step Efficiency): {avg_step_efficiency_micro:.4f}")
    else:
        print(f"Eff_macro (Macro-Average Step Efficiency): 0.0000")
        print(f"Eff_micro (Micro-Average Step Efficiency): 0.0000")


def main():
    parser = argparse.ArgumentParser(description="Calculate End-to-End Metrics (Efficiency Only).")
    parser.add_argument("--result", type=str, required=True, help="Path to the result JSONL file")
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        return

    print(f"Loading task results from: {result_path}")
    
    # Load Benchmark GT Steps
    # Automatically locate benchmark.csv relative to project root
    project_root = Path(__file__).resolve().parent.parent
    benchmark_path = project_root / "benchmark" / "benchmark.csv"
    gt_steps_map = load_benchmark_steps(benchmark_path)

    # 1. Load Task Data (Steps) from JSONL
    task_records = {}
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                tid = str(data.get('task_id'))
                if tid not in task_records:
                    task_records[tid] = []
                task_records[tid].append(data)
            except Exception as e:
                print(f"Error parsing line: {e}")

    task_data = {}
    
    # Determine mode from path logic improved
    # Path typically: .../results/model_name/agent_name/...
    # We try to infer agent_name from path components
    path_parts = str(result_path).replace("\\", "/").lower().split("/")
    
    agent_mode = "unknown"
    if "base" in path_parts:
        agent_mode = "base"
    elif "plan_and_solve" in path_parts:
        agent_mode = "plan_and_solve"
    elif "plan_and_react" in path_parts:
        agent_mode = "plan_and_react"
    elif "react" in path_parts:
        agent_mode = "react"
        
    print(f"Inferred Agent Mode: {agent_mode}")
    
    for tid, records in task_records.items():
        success_records = [r for r in records if r.get('status') == 'success']
        if success_records:
            final_record = success_records[-1]
        else:
            final_record = records[-1]
            
        # Fallback: check 'agent' field in JSON record if available
        record_agent = final_record.get('agent')
        current_mode = agent_mode
        if record_agent:
             # Map record agent name to our internal modes
             if record_agent == "base": current_mode = "base"
             elif record_agent == "react": current_mode = "react"
             elif record_agent == "plan_and_solve": current_mode = "plan_and_solve"
             elif record_agent == "plan_and_react": current_mode = "plan_and_react"

        history = final_record.get('history', [])
        actual_steps = count_actual_steps(history, current_mode)
        has_error = check_error_encountered(history)
        task_data[tid] = {"actual_steps": actual_steps, "has_error": has_error}

    # 2. Calculate Efficiency Metrics
    
    total_tasks = 0
    
    success_efficiencies = [] 
    success_gt_steps_total = 0
    success_actual_steps_total = 0 

    results_detail = []

    # Iterate over all tasks found in the result file
    # Sort by task ID for consistent output
    sorted_tids = sorted(task_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for task_id in sorted_tids:
        t_data = task_data[task_id]
        actual_steps = t_data["actual_steps"]
        has_error = t_data.get("has_error", False)
        gt_steps = gt_steps_map.get(task_id, 0)
        
        total_tasks += 1
        
        step_efficiency = 0.0
        
        # Determine if task was successful enough to calculate efficiency
        # Heuristic: if actual_steps > 0, we count it.
        is_valid_run = actual_steps > 0
        
        if is_valid_run:
            # Calculate Step Efficiency
            # Formula: gt / max(gt, actual)
            denominator = max(gt_steps, actual_steps)
            
            if denominator > 0:
                step_efficiency = gt_steps / denominator
                success_efficiencies.append(step_efficiency)
                
                success_gt_steps_total += gt_steps
                success_actual_steps_total += denominator
            
        results_detail.append({
            "task_id": task_id,
            "gt_steps": gt_steps,
            "actual_steps": actual_steps,
            "step_efficiency": step_efficiency
        })

    # 3. Compute Final Metrics
    
    print("\nEfficiency Evaluation Results Summary:")
    print("=" * 80)
    
    # Calculate success_count for total_stats
    total_success_count = len(success_efficiencies)

    # Total Metrics Construction
    total_stats = {
        "total": total_tasks,
        "success_count": total_success_count,
        "success_efficiencies": success_efficiencies,
        "success_gt_steps_total": success_gt_steps_total,
        "success_actual_steps_total": success_actual_steps_total
    }
    
    print_metrics("Overall Results", total_stats)
        
    print("=" * 80)
    
    # Report saving disabled per instruction
    # report_path = result_path.parent / f"{result_path.stem}_vlm_metrics.csv"
    # pd.DataFrame(results_detail).to_csv(report_path, index=False)
    # print(f"Detailed metrics saved to: {report_path}")

if __name__ == "__main__":
    main()