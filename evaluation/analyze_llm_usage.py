import json
import argparse
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Analyze Token Usage from LLM Logs")
    parser.add_argument("--log_file", type=str, default="logs/llm_usage.jsonl", help="Path to the log file (default: logs/llm_usage.jsonl)")
    parser.add_argument("--model", type=str, help="Filter by model name (e.g., 'gpt-4o', 'Qwen3-32B-AWQ'). If not specified, analyzes all models.")
    parser.add_argument("--result", type=str, help="Path to a benchmark result file (.jsonl) to automatically infer start/end times.")
    args = parser.parse_args()

    log_file = args.log_file
    target_model = args.model
    result_file = args.result

    valid_time_ranges = [] # List of (start_dt, end_dt) tuples

    # If result_file is provided, extract valid time ranges from it
    if result_file:
        if not os.path.exists(result_file):
            print(f"Error: Result file not found: {result_file}")
            return
            
        print(f"Reading valid time ranges from result file: {result_file}")
        try:
            # 1. Read and Deduplicate Tasks
            task_records = {} # {task_id: [records]}
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        tid = str(data.get("task_id"))
                        if tid not in task_records:
                            task_records[tid] = []
                        task_records[tid].append(data)
                    except:
                        continue
            
            # 2. Extract Valid Time Ranges from Final Records
            count_valid = 0
            for tid, records in task_records.items():
                # Priority: Success > Last Attempt
                success_records = [r for r in records if r.get("status") == "success"]
                if success_records:
                    final_record = success_records[-1]
                else:
                    final_record = records[-1]
                
                t_start = final_record.get("start_time")
                t_end = final_record.get("end_time")
                
                if t_start and t_end:
                    try:
                        dt_start = datetime.fromisoformat(t_start)
                        dt_end = datetime.fromisoformat(t_end)
                        valid_time_ranges.append((dt_start, dt_end))
                        count_valid += 1
                    except ValueError:
                        pass
            
            print(f"Identified {count_valid} valid task execution windows.")
            
        except Exception as e:
            print(f"Error reading result file: {e}")
            return

    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    total_ttft = 0.0
    total_speed = 0.0
    valid_ttft_count = 0
    valid_speed_count = 0

    print(f"Analyzing log file: {log_file}")
    if target_model:
        print(f"Filtering for model: {target_model}")
    else:
        print("Analyzing all models")
        
    if valid_time_ranges:
        print(f"Filtering by {len(valid_time_ranges)} valid task windows.")
    else:
        print("Warning: No valid time ranges found (or result file not provided). analyzing ALL logs.")
        
    print("-" * 40)

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # Filter by time if ranges exist
                    if valid_time_ranges:
                        record_time_str = record.get("timestamp")
                        if not record_time_str:
                            continue
                            
                        try:
                            record_time_dt = datetime.fromisoformat(record_time_str)
                            
                            # Check if time is within ANY valid range
                            is_valid_time = False
                            for (start, end) in valid_time_ranges:
                                if start <= record_time_dt <= end:
                                    is_valid_time = True
                                    break
                            
                            if not is_valid_time:
                                continue
                                
                        except ValueError:
                            continue


                    # Filter by model if specified
                    record_model = record.get("model")
                    if target_model and record_model != target_model:
                        continue

                    metrics = record.get("metrics", {})
                    
                    input_tokens = metrics.get("input_tokens", 0)
                    output_tokens = metrics.get("output_tokens", 0)
                    ttft = metrics.get("ttft", 0)
                    speed = metrics.get("speed", 0)
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_calls += 1

                    if ttft:
                        total_ttft += ttft
                        valid_ttft_count += 1
                    
                    if speed:
                        total_speed += speed
                        valid_speed_count += 1

                except json.JSONDecodeError:
                    continue

        if total_calls == 0:
            print("No matching records found.")
            return

        print(f"{'总调用次数 (Total Calls):':<40} {total_calls}")
        print(f"{'总输入Token (Total Input Tokens):':<40} {total_input_tokens:,}")
        print(f"{'总输出Token (Total Output Tokens):':<40} {total_output_tokens:,}")
        print(f"{'总Token (Total Tokens):':<40} {total_input_tokens + total_output_tokens:,}")
        
        if valid_ttft_count > 0:
            print(f"{'平均首字延迟 (Average TTFT):':<40} {total_ttft / valid_ttft_count:.4f} s")
        
        if valid_speed_count > 0:
            print(f"{'平均生成速度 (Average Speed):':<40} {total_speed / valid_speed_count:.2f} tokens/s")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()