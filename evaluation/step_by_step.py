import json
import os
import csv
import re
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
import argparse

# Tools that are considered optional/informational and can be skipped during evaluation
OPTIONAL_TOOLS = [
    "get_vector_info", 
    "get_raster_info",
    "get_csv_info",
    "get_nc_info",
    "get_graphml_info",
    "get_pkl_info"
]

# Plotting tools where output filename is strictly required (fixed final output)
PLOTTING_TOOLS = [
    "create_multilayer_map",
    "create_travel_time_network_map",
    "visualize_vector",
    "visualize_raster"
]

# Common file extensions used in this benchmark
PATH_EXTENSIONS = {
    ".tif", ".tiff", ".geojson", ".shp", ".csv", ".json", ".png", ".jpg",
    ".jpeg", ".pkl", ".npy", ".gpkg", ".nc", ".dbf", ".prj", ".shx"
}

def should_ignore_param(key: str, tool_name: str) -> bool:
    """
    Determine if a parameter should be ignored based on its name and the tool being used.
    Rules:
    1. Ignore if key is "title".
    2. Do NOT ignore path-related parameters; we compare them with variable renaming.
    3. Do NOT ignore "output_name"; for plotting tools it must match exactly.
    """
    if key == "title":
        return True

    return False

def _is_path_like(value: str) -> bool:
    """Heuristic to detect path-like strings."""
    if not isinstance(value, str):
        return False
    v = value.strip().replace("\\", "/").lower()
    if "/" in v:
        return True
    return any(v.endswith(ext) for ext in PATH_EXTENSIONS)

def _normalize_literal_string(value: str) -> str:
    """Normalize string literals; paths are normalized more aggressively."""
    if not isinstance(value, str):
        return value
    v = value.strip().replace("\\", "/")
    if v.startswith("./"):
        v = v[2:]
    if _is_path_like(v):
        return v.lower()
    return v

def _candidate_tokens(value: str) -> Set[str]:
    """Generate candidate tokens for matching output references."""
    if not isinstance(value, str):
        return set()
    v = _normalize_literal_string(value)
    tokens = {v}
    v_path = v.replace("\\", "/")
    if v_path.startswith("./"):
        v_path = v_path[2:]
    tokens.add(v_path)
    if "/" in v_path:
        tokens.add(os.path.basename(v_path))
    return {t for t in tokens if t}

def _normalize_output_token(value: str) -> str:
    """Normalize output_name tokens for variable mapping."""
    if not isinstance(value, str):
        return value
    return value.strip().lower()

def _resolve_output_ref(value: str, outputs: Set[str]) -> str:
    """Return the matched output token if the value references a prior output."""
    for token in _candidate_tokens(value):
        token_norm = token.lower()
        if token_norm in outputs:
            return token_norm
    return None

def _compare_values_with_mapping(
    pred: Any,
    gt: Any,
    tool_name: str,
    gt_outputs: Set[str],
    pred_outputs: Set[str],
    var_map: Dict[str, str],
    key: str = None,
    output_dir: str = None
) -> Tuple[bool, Dict[str, str]]:
    """
    Compare two values with variable renaming support and return (match, var_map).
    """
    if type(pred) != type(gt):
        return False, var_map

    if isinstance(pred, dict):
        if set(pred.keys()) != set(gt.keys()):
            return False, var_map
        for k in gt.keys():
            ok, var_map = _compare_values_with_mapping(
                pred[k], gt[k], tool_name, gt_outputs, pred_outputs, var_map, key=k, output_dir=output_dir
            )
            if not ok:
                return False, var_map
        return True, var_map

    if isinstance(pred, list):
        if len(pred) != len(gt):
            return False, var_map

        used = [False] * len(pred)

        def backtrack(i: int, current_map: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
            if i == len(gt):
                return True, current_map
            for j in range(len(pred)):
                if used[j]:
                    continue
                map_copy = dict(current_map)
                ok, map_copy = _compare_values_with_mapping(
                    pred[j], gt[i], tool_name, gt_outputs, pred_outputs, map_copy, key=key, output_dir=output_dir
                )
                if ok:
                    used[j] = True
                    success, final_map = backtrack(i + 1, map_copy)
                    if success:
                        return True, final_map
                    used[j] = False
            return False, current_map

        return backtrack(0, var_map)

    if isinstance(pred, str):
        # Output name handling: allow renaming for non-plotting tools
        if key == "output_name" and isinstance(gt, str):
            if tool_name in PLOTTING_TOOLS:
                return _normalize_literal_string(pred) == _normalize_literal_string(gt), var_map
            
            # STRICT CHECK: Even if we allow renaming, the predicted output file MUST exist if output_dir is provided
            if output_dir:
                 pred_path_obj = Path(pred)
                 file_exists = False
                 # 1. Check in moved output_dir
                 p_filename = pred_path_obj.name
                 if (Path(output_dir) / p_filename).exists():
                     file_exists = True
                 elif len(pred_path_obj.parts) > 1 and pred_path_obj.parts[0].lower() == 'output':
                     if Path(output_dir).joinpath(*pred_path_obj.parts[1:]).exists():
                         file_exists = True
                 
                 # 2. Check in CWD (unlikely for output_name, but for consistency)
                 if not file_exists and pred_path_obj.exists():
                     file_exists = True
                 
                 if not file_exists:
                     return False, var_map

            gt_token = _normalize_output_token(gt)
            pred_token = _normalize_output_token(pred)
            existing = var_map.get(gt_token)
            if existing:
                return existing == pred_token, var_map
            if pred_token in var_map.values():
                return False, var_map
            var_map[gt_token] = pred_token
            return True, var_map

        # Variable reference handling (inputs referencing prior outputs)
        gt_ref = _resolve_output_ref(gt, gt_outputs)
        pred_ref = _resolve_output_ref(pred, pred_outputs)
        if gt_ref or pred_ref:
            if not gt_ref or not pred_ref:
                return False, var_map
            existing = var_map.get(gt_ref)
            if existing:
                return existing == pred_ref, var_map
            if pred_ref in var_map.values():
                return False, var_map
            var_map[gt_ref] = pred_ref
            return True, var_map

        # Path validation logic: Check if file exists for *_path parameters
        if key and key.endswith("_path") and output_dir:
            # Construct potential file path
            pred_path_obj = Path(pred)
            file_exists = False
            
            # 1. Check in moved output_dir (for generated outputs)
            # Try filename match
            p_filename = pred_path_obj.name
            if (Path(output_dir) / p_filename).exists():
                file_exists = True
            # Try preserving relative structure (strip 'output' prefix)
            elif len(pred_path_obj.parts) > 1 and pred_path_obj.parts[0].lower() == 'output':
                 if Path(output_dir).joinpath(*pred_path_obj.parts[1:]).exists():
                     file_exists = True
            
            # 2. Check in current working directory (for original dataset inputs)
            if not file_exists:
                if pred_path_obj.exists():
                    file_exists = True
            
            if not file_exists:
                return False, var_map

        # Literal string compare (with path normalization)
        return _normalize_literal_string(pred) == _normalize_literal_string(gt), var_map

    return pred == gt, var_map

def recursive_filter(args: Any, tool_name: str) -> Any:
    """
    Recursively filter arguments (dict or list) based on ignore rules.
    """
    if isinstance(args, dict):
        return {
            k: recursive_filter(v, tool_name)
            for k, v in args.items()
            if not should_ignore_param(k, tool_name)
        }
    elif isinstance(args, list):
        return [recursive_filter(item, tool_name) for item in args]
    else:
        return args

def unordered_compare(pred: Any, gt: Any) -> bool:
    """
    Compare two objects allowing for unordered lists.
    Recursively compares dicts and lists.
    """
    if type(pred) != type(gt):
        return False
        
    if isinstance(pred, dict):
        if set(pred.keys()) != set(gt.keys()):
            return False
        return all(unordered_compare(pred[k], gt[k]) for k in pred)
        
    if isinstance(pred, list):
        if len(pred) != len(gt):
            return False
        # Try to match elements (greedy match for unordered lists)
        # Note: This is O(N^2) which is fine for small argument lists
        gt_copy = list(gt)
        for p_item in pred:
            matched = False
            for i, g_item in enumerate(gt_copy):
                if unordered_compare(p_item, g_item):
                    gt_copy.pop(i)
                    matched = True
                    break
            if not matched:
                return False
        return True
        
    return pred == gt


def load_benchmark(path):
    data = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['ID']] = {
                'id': row['ID'],
                'gt_length': int(row['Toolchain Length']),
                'gt_toolchain': json.loads(row['Toolchain JSON']),
            }
    return data

def extract_tool_calls(history, mode="react"):
    tool_calls = []
    # Iterate through history
    for msg in history:
        if msg['role'] == 'assistant':
            content = ""
            if isinstance(msg['content'], str):
                content = msg['content']
            elif isinstance(msg['content'], list):
                for part in msg['content']:
                    if part.get('type') == 'text':
                        content += part.get('text', '')
            
            # 1. Check for OpenAI tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tc in msg['tool_calls']:
                    if 'function' in tc:
                         try:
                            args = json.loads(tc['function']['arguments'])
                            tool_calls.append({"name": tc['function']['name'], "arguments": args})
                         except:
                            pass
                continue

            # 2. Extract based on mode
            if mode == "base":
                # Direct/BaseAgent style "<tool_call>...</tool_call>"
                # Matches <tool_call>\n{...}\n</tool_call>
                xml_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.MULTILINE | re.DOTALL)
                for json_str in xml_matches:
                    try:
                        data = json.loads(json_str)
                        if 'name' in data and 'arguments' in data:
                            tool_calls.append({"name": data['name'], "arguments": data['arguments']})
                    except:
                        pass
            elif mode == "plan_and_solve":
                # PlanAndSolve style: JSON plan in content
                try:
                    # Logic adapted from agents/plan_and_solve.py
                    plan_json_str = ""
                    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                    if json_match:
                        plan_json_str = json_match.group(1)
                    else:
                        json_match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
                        if json_match:
                            plan_json_str = json_match.group(1)
                        else:
                            plan_json_str = content

                    # Clean up JSON string
                    plan_json_str = re.sub(r'//.*', '', plan_json_str)
                    plan_json_str = re.sub(r'/\*.*?\*/', '', plan_json_str, flags=re.DOTALL)
                    
                    if not plan_json_str.strip():
                        continue

                    plan_data = json.loads(plan_json_str)
                    plan_steps = []
                    
                    if isinstance(plan_data, dict) and "plan" in plan_data:
                        plan_steps = plan_data["plan"]
                    elif isinstance(plan_data, list):
                        plan_steps = plan_data
                    elif isinstance(plan_data, dict):
                        plan_steps = [plan_data]
                    
                    # Validate and extract
                    current_tool_calls = []
                    for step in plan_steps:
                        if isinstance(step, dict):
                            tool_name = step.get("tool") or step.get("name")
                            if tool_name:
                                current_tool_calls.append({
                                    "name": tool_name,
                                    "arguments": step.get("parameters", {}) or step.get("arguments", {})
                                })
                    
                    if current_tool_calls:
                        # Found the plan, return immediately as plan_and_solve usually outputs one plan
                        return current_tool_calls
                except:
                    pass
            else:
                # ReAct style "Action: {...}" in content
                matches = re.findall(r'(?m)^Action:\s*(\{.*\})', content, re.MULTILINE | re.DOTALL)
                for json_str in matches:
                    try:
                        data = json.loads(json_str)
                        # Check for "name" and "arguments"
                        if 'name' in data and 'arguments' in data:
                            tool_calls.append({"name": data['name'], "arguments": data['arguments']})
                    except:
                        pass

    return tool_calls

def process_raw_results(result_path: str, benchmark_path: str) -> List[Dict]:
    """
    Process raw result JSONL and benchmark CSV to generate evaluation data
    """
    print(f"Loading benchmark from {benchmark_path}...")
    benchmark_data = load_benchmark(benchmark_path)
    print(f"Loaded {len(benchmark_data)} benchmark entries.")

    results = []
    
    print(f"Processing results from {result_path}...")
    
    # Determine mode from path (Initial Guess)
    path_parts = str(result_path).replace("\\", "/").lower().split("/")
    
    # Infer Output Directory
    output_dir = None
    try:
        p = Path(result_path).resolve()
        parts = p.parts
        # Find 'results' part index
        results_idx = -1
        for i, part in enumerate(parts):
            if part.lower() == 'results':
                results_idx = i
                break
        
        if results_idx != -1 and results_idx < len(parts) - 2:
            # Reconstruct path: .../output_results/model/agent/output
            # Structure: .../results/model/agent/filename.jsonl
            # Target:    .../output_results/model/agent/output
            base_parts = list(parts[:results_idx])
            sub_parts = list(parts[results_idx+1:-1]) # model/agent
            
            new_parts = base_parts + ["output_results"] + sub_parts + ["output"]
            output_dir = str(Path(*new_parts))
            print(f"Inferred Output Directory: {output_dir}")
            if not os.path.exists(output_dir):
                print(f"Warning: Output directory does not exist: {output_dir}")
    except Exception as e:
        print(f"Error inferring output directory: {e}")

    global_mode = "react" # Default
    if "base" in path_parts:
        global_mode = "base"
    elif "plan_and_solve" in path_parts:
        global_mode = "plan_and_solve"
    elif "plan_and_react" in path_parts:
        global_mode = "react" # PlanAndReact uses ReAct style extraction
    elif "react" in path_parts:
        global_mode = "react"
        
    print(f"Inferred Global Agent Mode: {global_mode}")
    
    # Read all records first
    task_records = {} # {task_id: [records]}
    with open(result_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
                task_id = str(entry.get('task_id'))
                if task_id not in task_records:
                    task_records[task_id] = []
                task_records[task_id].append(entry)
            except Exception as e:
                print(f"Error processing line {i}: {e}")

    # Deduplicate and Select
    final_entries = []
    sorted_tids = sorted(task_records.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for tid in sorted_tids:
        records = task_records[tid]
        success_records = [r for r in records if r.get('status') == 'success']
        
        if success_records:
            final_entries.append(success_records[-1])
        else:
            final_entries.append(records[-1])

    for entry in final_entries:
        try:
            task_id = str(entry.get('task_id'))
            bench_entry = benchmark_data.get(task_id)
            
            if not bench_entry:
                print(f"Warning: Task ID {task_id} not found in benchmark.")
                continue
            
            # Determine specific mode for this entry
            record_agent = entry.get('agent')
            current_mode = global_mode
            
            if record_agent:
                if record_agent == "base":
                    current_mode = "base"
                elif record_agent == "plan_and_solve":
                    current_mode = "plan_and_solve"
                elif record_agent in ["react", "plan_and_react"]:
                    current_mode = "react"
            
            history = entry.get('history', [])
            pred_tools = extract_tool_calls(history, current_mode)
            
            results.append({
                "id": bench_entry['id'],
                "query": entry.get('query', ''),
                "gt_toolchain": bench_entry['gt_toolchain'],
                "pred_toolchain": pred_tools,
                "gt_length": bench_entry['gt_length'],
                "output_dir": output_dir
            })
        except Exception as e:
            print(f"Error processing entry {task_id}: {e}")

    print(f"Processed {len(results)} entries.")
    return results

def extract_tool_names(tool_calls: List[Dict]) -> List[str]:
    """Extract tool names from tool calls list"""
    if not tool_calls:
        return []
    # Handle case where tool_calls might be None or not a list
    if not isinstance(tool_calls, list):
        return []
    return [call.get("name") or call.get("tool", "") for call in tool_calls]

def calculate_tool_selection_metrics(pred_tools: List[Dict], gt_tools: List[Dict]) -> dict:
    """
    Check tool usage statistics (Recall, Precision, F1) ignoring order
    """
    expected_tools = extract_tool_names(gt_tools)
    actual_tools = extract_tool_names(pred_tools)
    
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    matched_tools = expected_set.intersection(actual_set)
    
    # Recall (original score)
    if len(expected_set) > 0:
        recall = len(matched_tools) / len(expected_set)
    else:
        recall = 1.0 if len(actual_set) == 0 else 0.0
        
    # Precision
    if len(actual_set) > 0:
        precision = len(matched_tools) / len(actual_set)
    else:
        precision = 1.0 if len(expected_set) == 0 else 0.0
        
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "key": "TAO",
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "details": {
            "matched_tools": len(matched_tools),
            "total_expected": len(expected_set),
            "total_predicted": len(actual_set),
            "matched_tool_names": list(matched_tools)
        }
    }

def calculate_relative_sequence_match(pred_tools: List[Dict], gt_tools: List[Dict]) -> dict:
    """
    Check if all expected tool calls are contained (considering order, but allowing other tools in between)
    """
    expected_tools = extract_tool_names(gt_tools)
    actual_tools = extract_tool_names(pred_tools)
    
    if not expected_tools:
        return {
            "score": 1.0,
            "key": "relative_sequence",
            "details": {"matched_in_order": 0, "total_expected": 0}
        }
    
    matched_count = 0
    actual_iter = iter(actual_tools)
    
    for expected_tool in expected_tools:
        found = False
        for actual_tool in actual_iter:
            if actual_tool == expected_tool:
                matched_count += 1
                found = True
                break
        if not found:
            break
    
    score = matched_count / len(expected_tools)
    
    return {
        "score": score, 
        "key": "TIO",
        "details": {
            "matched_in_order": matched_count,
            "total_expected": len(expected_tools)
        }
    }

def calculate_trajectory_exact_match(pred_tools: List[Dict], gt_tools: List[Dict]) -> dict:
    """
    Calculate trajectory step-wise score (strict order matching from the beginning)
    """
    expected_tools = extract_tool_names(gt_tools)
    actual_tools = extract_tool_names(pred_tools)

    if not expected_tools:
        return {"score": 1.0 if not actual_tools else 0.0, "key": "trajectory_exact"}

    correct_steps = 0
    min_length = min(len(expected_tools), len(actual_tools))

    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            correct_steps += 1
        else:
            break

    score = correct_steps / len(expected_tools) if len(expected_tools) > 0 else 0

    return {
        "score": score,
        "key": "TEM",
        "details": {
            "correct_steps": correct_steps,
            "total_expected": len(expected_tools)
        }
    }

def calculate_parameter_accuracy(pred_tools: List[Dict], gt_tools: List[Dict], output_dir: str = None) -> dict:
    """
    Check the accuracy of tool call parameters using a Two-Pass Approach:
    1. Backward Alignment: Find the best matching prediction for each GT step (handling retries).
    2. Forward Evaluation: Evaluate parameters in order to correctly handle variable mapping and flow.
    """
    # 1. Include plotting tools in parameter accuracy check
    pred_tools_filtered = pred_tools
    gt_tools_filtered = gt_tools

    if not gt_tools_filtered:
        return {
            "score": 1.0,
            "key": "parameter_accuracy",
            "details": {"matched_steps": 0, "total_expected_steps": 0}
        }
    
    total_expected_steps = len(gt_tools_filtered)
    
    # --- Pass 1: Backward Alignment ---
    # Map gt_index -> pred_index
    alignment = {}
    pred_cursor = len(pred_tools_filtered) - 1

    for i in range(total_expected_steps - 1, -1, -1):
        exp_call = gt_tools_filtered[i]
        exp_name = exp_call.get("name") or exp_call.get("tool")
        
        # Search backwards in pred_tools from pred_cursor
        match_idx = -1
        for idx in range(pred_cursor, -1, -1):
            p_call = pred_tools_filtered[idx]
            p_name = p_call.get("name") or p_call.get("tool")
            
            if p_name == exp_name:
                match_idx = idx
                break
        
        if match_idx != -1:
            alignment[i] = match_idx
            pred_cursor = match_idx - 1

    # --- Pass 2: Forward Evaluation ---
    matched_steps = 0
    parameter_details = []

    # Variable mapping and output tracking (shared across steps)
    gt_outputs: Set[str] = set()
    pred_outputs: Set[str] = set()
    var_map: Dict[str, str] = {}

    for i in range(total_expected_steps):
        exp_call = gt_tools_filtered[i]
        exp_name = exp_call.get("name") or exp_call.get("tool")
        exp_args = exp_call.get("arguments") or exp_call.get("input") or {}
        
        if isinstance(exp_args, str):
            try:
                exp_args = json.loads(exp_args)
            except:
                pass

        current_step_correct = False
        actual_tool_name = None
        found_match = i in alignment
        
        if found_match:
            pred_idx = alignment[i]
            final_pred_call = pred_tools_filtered[pred_idx]
            actual_tool_name = final_pred_call.get("name") or final_pred_call.get("tool")

            # Parameter Comparison
            pred_args = final_pred_call.get("arguments") or final_pred_call.get("input") or {}
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except:
                    pass

            pred_args_filtered = recursive_filter(pred_args, exp_name)
            exp_args_filtered = recursive_filter(exp_args, exp_name)

            # Special rule for plotting tools: only compare output_name
            if exp_name in PLOTTING_TOOLS:
                pred_args_filtered = {"output_name": pred_args_filtered.get("output_name")} if isinstance(pred_args_filtered, dict) else {}
                exp_args_filtered = {"output_name": exp_args_filtered.get("output_name")} if isinstance(exp_args_filtered, dict) else {}

            args_match, var_map = _compare_values_with_mapping(
                pred_args_filtered,
                exp_args_filtered,
                exp_name,
                gt_outputs,
                pred_outputs,
                var_map,
                output_dir=output_dir
            )

            if args_match:
                current_step_correct = True
            
            # Update variable flow (Outputs)
            gt_out = exp_args_filtered.get("output_name") if isinstance(exp_args_filtered, dict) else None
            pred_out = pred_args_filtered.get("output_name") if isinstance(pred_args_filtered, dict) else None
            if isinstance(gt_out, str):
                gt_outputs.add(_normalize_output_token(gt_out))
            if isinstance(pred_out, str):
                pred_outputs.add(_normalize_output_token(pred_out))

        call_detail = {
            "step": i + 1,
            "expected_tool": exp_name,
            "actual_tool": actual_tool_name,
            "name_match": found_match,
            "args_match": current_step_correct,
            "is_correct": current_step_correct
        }
        
        if current_step_correct:
            matched_steps += 1
        
        parameter_details.append(call_detail)
            
    score = matched_steps / total_expected_steps
    
    return {
        "score": score,
        "key": "PEA",
        "details": {
            "matched_steps": matched_steps,
            "total_expected_steps": total_expected_steps,
            "details": parameter_details
        }
    }

def filter_optional_tools(tool_calls: List[Dict]) -> List[Dict]:
    """
    Filter out optional tools from the tool calls list.
    """
    if not tool_calls:
        return []
    
    filtered_calls = []
    for call in tool_calls:
        tool_name = call.get("name") or call.get("tool", "")
        if tool_name not in OPTIONAL_TOOLS:
            filtered_calls.append(call)
            
    return filtered_calls

def evaluate_single_entry(entry: Dict) -> Dict:
    """
    Perform evaluation for a single entry
    """
    gt_tools = entry.get("gt_toolchain", [])
    pred_tools = entry.get("pred_toolchain", [])
    
    # Handle case where gt_toolchain might be a string (if not parsed yet)
    if isinstance(gt_tools, str):
        try:
            gt_tools = json.loads(gt_tools)
        except:
            print(f"Warning: Could not parse gt_toolchain for ID {entry.get('id')}")
            gt_tools = []

    # Apply Optional Tools Filtering
    # We filter both Ground Truth and Prediction to focus on core logic comparison
    gt_tools_filtered = filter_optional_tools(gt_tools)
    pred_tools_filtered = filter_optional_tools(pred_tools)

    results = {}
    results["TAO"] = calculate_tool_selection_metrics(pred_tools_filtered, gt_tools_filtered)
    results["TIO"] = calculate_relative_sequence_match(pred_tools_filtered, gt_tools_filtered)
    results["TEM"] = calculate_trajectory_exact_match(pred_tools_filtered, gt_tools_filtered)
    
    output_dir = entry.get("output_dir")
    results["PEA"] = calculate_parameter_accuracy(pred_tools_filtered, gt_tools_filtered, output_dir=output_dir)
    
    return results

def init_metrics_stats():
    return {
        "total_entries": 0,
        "metrics": {
            "TAO": {"recall_total": 0, "precision_total": 0, "f1_total": 0, "count": 0},
            "TIO": {"total": 0, "count": 0},
            "TEM": {"total": 0, "count": 0},
            "PEA": {"total": 0, "count": 0}
        }
    }

def update_metrics_stats(stats, eval_result):
    stats["total_entries"] += 1
    for metric, res in eval_result.items():
        if metric == "TAO":
            m_stats = stats["metrics"][metric]
            m_stats["recall_total"] += res["recall"]
            m_stats["precision_total"] += res["precision"]
            m_stats["f1_total"] += res["f1"]
            m_stats["count"] += 1
        elif metric in stats["metrics"]:
            stats["metrics"][metric]["total"] += res["score"]
            stats["metrics"][metric]["count"] += 1

def print_metrics_report(title, stats):
    print(f"\n--- {title} (Count: {stats['total_entries']}) ---")
    metrics = stats["metrics"]
    
    # 1. TAO (Tools-Any-Order)
    if "TAO" in metrics:
        s = metrics["TAO"]
        if s["count"] > 0:
            avg_f1 = s["f1_total"] / s["count"]
            avg_p = s["precision_total"] / s["count"]
            avg_r = s["recall_total"] / s["count"]
            print(f"TAO (Tools-Any-Order): F1: {avg_f1:.4f} | Precision: {avg_p:.4f} | Recall: {avg_r:.4f}")
        else:
            print(f"TAO (Tools-Any-Order): F1: 0.0000 | Precision: 0.0000 | Recall: 0.0000")

    # 2. TIO (Tools-In-Order)
    if "TIO" in metrics:
        s = metrics["TIO"]
        avg = s["total"] / s["count"] if s["count"] > 0 else 0
        print(f"TIO (Tools-In-Order): {avg:.4f}")

    # 3. TEM (Tool-Exact-Match)
    if "TEM" in metrics:
        s = metrics["TEM"]
        avg = s["total"] / s["count"] if s["count"] > 0 else 0
        print(f"TEM (Tool-Exact-Match): {avg:.4f}")

    # 4. PEA (Parameter Execution Accuracy)
    if "PEA" in metrics:
        s = metrics["PEA"]
        avg = s["total"] / s["count"] if s["count"] > 0 else 0
        print(f"PEA (Parameter Execution Accuracy): {avg:.4f}")

def run_evaluation(data: List[Dict]) -> Dict:
    """
    Run evaluation on the list of data entries
    """
    all_results = {}
    
    total_stats = init_metrics_stats()
    
    print(f"Evaluating {len(data)} entries...")
    
    for entry in data:
        entry_id = entry.get("id")
        eval_result = evaluate_single_entry(entry)
        all_results[entry_id] = eval_result
        
        # Update Total
        update_metrics_stats(total_stats, eval_result)
                
    # Calculate averages
    print("\nEvaluation Results Summary:")
    print("=" * 80)
    
    print_metrics_report("Overall Results", total_stats)
            
    print("=" * 80)
    
    return {
        "summary": total_stats,
        "details": all_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Step-by-Step Evaluation")
    
    parser.add_argument("--benchmark", type=str, default=r"benchmark/benchmark.csv", help="Path to benchmark CSV file")
    parser.add_argument("--result", type=str, required=True, help="Path to raw result JSONL file")
    
    args = parser.parse_args()
    
    data = []
    if args.result and args.benchmark:
        if os.path.exists(args.result) and os.path.exists(args.benchmark):
            data = process_raw_results(args.result, args.benchmark)
        else:
            print(f"Error: Result file '{args.result}' or Benchmark file '{args.benchmark}' not found.")
            exit(1)
    
    if data:
        run_evaluation(data)