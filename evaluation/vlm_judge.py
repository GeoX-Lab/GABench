import os
import json
import yaml
import base64
import pandas as pd
from pathlib import Path
from openai import OpenAI
import re
import glob
import io
import httpx
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load config
current_dir = Path(__file__).parent
project_root = current_dir.parent
load_dotenv(project_root / ".env")
config_path = project_root / "config.yaml"

with open(config_path, "r") as f:
    # Expand environment variables in config
    raw_config = os.path.expandvars(f.read())
    config = yaml.safe_load(raw_config)

eval_config = config.get("evaluation", {})
JUDGE_MODEL = eval_config.get("judge_model", "gpt-4o")

# Fixed paths as per instruction
DATASET_RESULT_DIR = project_root / "dataset" / "result"
OUTPUT_DIR = project_root / config.get("output_dir", "output")
BENCHMARK_FILE = project_root / "benchmark" / "benchmark.csv"

# Initialize OpenAI client from evaluation config
api_key = eval_config.get("api_key")
base_url = eval_config.get("base_url")

if not api_key or not base_url:
    # Fallback to looking up the model in llm config if specific auth not provided
    print(f"Auth not found in evaluation section, trying to find model {JUDGE_MODEL} in llm section...")
    llm_config = config.get("llm", {}).get(JUDGE_MODEL)
    if llm_config:
        api_key = llm_config.get("api_key")
        base_url = llm_config.get("base_url")

if not api_key:
    raise ValueError("API Key not found in evaluation config or llm config")

# Configure httpx client to ignore SSL errors and system proxies (fixes connection hanging)
http_client = httpx.Client(verify=False, trust_env=False)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=http_client
)

def stitch_images(img1_path, img2_path, label1="Ground Truth", label2="Prediction"):
    """
    Stitch two images side-by-side with labels.
    """
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Convert to RGB
        if img1.mode != 'RGB': img1 = img1.convert('RGB')
        if img2.mode != 'RGB': img2 = img2.convert('RGB')

        # Resize img2 to match img1's height while maintaining aspect ratio
        if img1.height != img2.height:
            aspect_ratio = img2.width / img2.height
            new_height = img1.height
            new_width = int(new_height * aspect_ratio)
            # Use appropriate resampling filter
            resample = getattr(Image, 'Resampling', Image).LANCZOS
            img2 = img2.resize((new_width, new_height), resample)

        # Layout dimensions
        gap = 20
        header_height = 50
        total_width = img1.width + img2.width + gap
        total_height = img1.height + header_height

        # Create new white image
        combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))

        # Paste images
        combined_img.paste(img1, (0, header_height))
        combined_img.paste(img2, (img1.width + gap, header_height))

        # Draw labels
        draw = ImageDraw.Draw(combined_img)
        
        # Try to use a font
        try:
            # Try arial on Windows
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            try:
                # Fallback to default but it's small, so maybe try load_default
                font = ImageFont.load_default()
            except:
                font = None

        # Center text roughly
        # If font is loaded (even default), use it.
        if font:
            draw.text((10, 5), label1, fill=(0, 0, 0), font=font)
            draw.text((img1.width + gap + 10, 5), label2, fill=(0, 0, 0), font=font)
        else:
            # Fallback if really no font (unlikely with PIL)
            pass

        return combined_img
    except Exception as e:
        print(f"Error stitching images: {e}")
        return None

def encode_pil_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def find_file(filename, search_dir):
    # Direct check
    p = search_dir / filename
    if p.exists():
        return p
    # Recursive check
    files = list(search_dir.rglob(filename))
    if files:
        return files[0]
    return None

def evaluate_value_metric(rule_str, output_dir=None):
    # Logic from end_to_end.py adapted
    # Format: CHECK:TYPE:filename:key:condition
    # TYPE: JSON_VALUE | CSV_VALUE
    try:
        parts = rule_str.split(':')
        if len(parts) < 5:
            return 0.0, "Invalid rule format"
        
        check_type = parts[1]
        filename_pattern = parts[2]
        key = parts[3]
        condition = parts[4]
        
        # Use provided output_dir or fallback to global
        search_dir = output_dir if output_dir else OUTPUT_DIR
        target_file = find_file(filename_pattern, search_dir)
        
        if not target_file:
            return 0.0, f"File {filename_pattern} not found"
            
        val = None
        
        if check_type == "JSON_VALUE":
            with open(target_file, 'r') as f:
                data = json.load(f)
            val = data.get(key)
        elif check_type == "CSV_VALUE":
            # For CSV, we assume the key is the column name and we take the first row's value
            try:
                df = pd.read_csv(target_file)
                if key not in df.columns:
                    return 0.0, f"Column {key} not found in CSV"
                if len(df) == 0:
                    return 0.0, "CSV is empty"
                val = df.iloc[0][key]
            except Exception as e:
                return 0.0, f"CSV Read Error: {e}"
        else:
            return 0.0, f"Unknown check type: {check_type}"

        if val is None:
            return 0.0, f"Key {key} not found or value is None"
            
        # Evaluation logic
        passed = False
        if condition == "EXISTS":
            passed = True
        elif condition.lower() == "==true":
            passed = (val is True)
        elif condition.lower() == "==false":
            passed = (val is False)
        else:
            # Numeric comparison
            # Handle string-numeric conversion if needed
            if isinstance(val, str):
                try:
                    val = float(val)
                except:
                    pass # Keep as string if not float

            match = re.match(r'([<>=!]+)([\d\.]+)', condition)
            # Support simple range syntax: val-val
            range_match = re.match(r'([\d\.]+)-([\d\.]+)', condition)
            
            if range_match:
                 min_val = float(range_match.group(1))
                 max_val = float(range_match.group(2))
                 if isinstance(val, (int, float)):
                     passed = min_val <= val <= max_val
            elif match:
                op = match.group(1)
                threshold = float(match.group(2))
                if isinstance(val, (int, float)):
                    if op == '>': passed = val > threshold
                    elif op == '>=': passed = val >= threshold
                    elif op == '<': passed = val < threshold
                    elif op == '<=': passed = val <= threshold
                    elif op == '==': passed = val == threshold
                    elif op == '!=': passed = val != threshold
            else:
                 # String exact match fallback
                 passed = str(val) == condition
        
        return (1.0 if passed else 0.0), f"Value {val} condition {condition}"

    except Exception as e:
        return 0.0, str(e)

def calculate_categorical_score(data_cat: str, style_cat: str) -> int:
    """
    根据 VLM 返回的分类计算最终得分。
    """
    # 评分矩阵
    score_matrix = {
        # 数据完美
        ("EXACT", "PERFECT"): 100,
        ("EXACT", "MINOR_DIFF"): 90,   # 允许细微渲染差异
        ("EXACT", "MISMATCH"): 60,     # 数据对，但风格违规（如颜色错）
        
        # 数据可接受
        ("ACCEPTABLE", "PERFECT"): 85,
        ("ACCEPTABLE", "MINOR_DIFF"): 75,
        ("ACCEPTABLE", "MISMATCH"): 50, # 勉强及格

        # 数据严重错误
        ("MAJOR_ERROR", "PERFECT"): 30,
        ("MAJOR_ERROR", "MINOR_DIFF"): 30,
        ("MAJOR_ERROR", "MISMATCH"): 20,
        
        # 失败
        ("FAILURE", "NA"): 0,
        ("FAILURE", "MISMATCH"): 0
    }
    
    # 标准化输入（转大写，处理可能的 None）
    d_key = str(data_cat).upper().strip()
    s_key = str(style_cat).upper().strip()
    
    # 兜底逻辑：如果 VLM 输出的类别不在字典里
    if d_key == "FAILURE": return 0
    if d_key == "MAJOR_ERROR": return 30
    
    # 查找分数，默认给 0
    # 处理一些可能的 key 组合缺失情况
    return score_matrix.get((d_key, s_key), 0)

def evaluate_image(gt_path, pred_path, task_instruction="N/A", style_instruction="N/A"):
    try:
        # Stitch images side-by-side
        combined_img = stitch_images(gt_path, pred_path, label1="Ground Truth (Left)", label2="Prediction (Right)")
        if combined_img is None:
             return 0, "Failed to stitch images"
             
        img_b64 = encode_pil_image(combined_img)
        
        prompt = f"""
            You are a Senior GIS QA Engineer evaluating an AI-generated map.
            Your goal is to compare a **Prediction Image (Right)** against a **Ground Truth Image (Left)** based on the provided Task Instructions.

            <Task_Context>
            - **User Instruction**: "{task_instruction}"
            - **Required Style**: "{style_instruction}"
            </Task_Context>

            <Input_Description>
            The input image is a side-by-side comparison:
            - **LEFT**: Ground Truth (GT) - The standard correct answer.
            - **RIGHT**: Prediction (Pred) - The agent's output.
            </Input_Description>

            <Evaluation_Steps>
            You must perform the evaluation in two steps.

            **STEP 1: Visual Analysis (Chain of Thought)**
            First, analyze the images and write down your observations inside <analysis> tags. You must explicitly check the following specific GIS features:
            1. **Geographic Extent (BBox)**: Does the Pred cover the exact same geographic region as the GT? Look at the coastlines, borders, and shapes.
            2. **Data Distribution**: Are the high/low value areas (hotspots) in the same location?
            3. **Geometry/Features**: Are the vector shapes (polygons, lines, points) consistent with the GT?
            4. **Style & Legend**: Does the color scheme (colormap) match the GT? Is the legend present and correct?
            5. **Major Artifacts**: Is the map blank, broken, or containing error text?

            *Note: Ignore minor pixel-level differences, anti-aliasing issues, or slight text positioning shifts. Focus on SEMANTIC correctness.*

            **STEP 2: Scoring & JSON Output**
            Based *strictly* on your analysis in Step 1, classify the result using the criteria below.

            ### Dimension 1: Data & Spatial Accuracy
            - "EXACT": The map shows the same region and data. Visually, it conveys the exact same information as GT. (Allows for tiny rendering differences).
            - "ACCEPTABLE": Correct region and general data pattern, but has minor missing features (e.g., missing specific labels) or slight resolution drop.
            - "MAJOR_ERROR": Wrong geographic region, wrong data values (e.g., hotspot in wrong place), or hallucinated features.
            - "FAILURE": Blank image, error message, or completely unintelligible noise.

            ### Dimension 2: Style Adherence
            - "PERFECT": Follows the requested style (e.g., Heatmap/Choropleth) AND uses the same color palette/ramp as GT.
            - "MINOR_DIFF": Follows the general style (e.g., correct chart type) but uses a different color scheme or font than GT.
            - "MISMATCH": Violates the requested style (e.g., asked for Heatmap, drew Scatter) or uses a misleading visualization.
            - "NA": Use this ONLY if Data Accuracy is "FAILURE".

            </Evaluation_Steps>

            <Output_Format>
            Your final output must contain the analysis followed by the JSON object.
            Example:
            <analysis>
            The GT shows a heatmap of population in Tokyo using a red gradient.
            The Pred also shows Tokyo with the same coastline shape.
            The hotspots are in the same location (central Tokyo).
            The color ramp is identical (white to red).
            Conclusion: The images are semantically identical.
            </analysis>
            ```json
            {{
                "data_category": "EXACT",
                "style_category": "PERFECT",
                "reason": "The prediction accurately renders the population heatmap for Tokyo. The geographic extent, data distribution, and styling match the Ground Truth perfectly."
            }}
        """
        
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
            # response_format={"type": "json_object"}  # Conflict with <analysis> tag in prompt
        )
        
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except Exception:
            content_str = content.strip()
            content_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", content_str, flags=re.IGNORECASE | re.DOTALL)
            match = re.search(r"\{[\s\S]*\}", content_str)
            if not match:
                raise
            result = json.loads(match.group(0))
        
        # Parse classification and calculate score
        data_cat = result.get("data_category", "FAILURE")
        style_cat = result.get("style_category", "NA")
        reason = result.get("reason", "No reason provided")
        
        final_score = calculate_categorical_score(data_cat, style_cat)
        
        # Add category labels to reason for further analysis
        full_reason = f"[{data_cat}/{style_cat}] {reason}"
        
        return final_score, full_reason

    except Exception as e:
        return 0, f"VLM Error: {str(e)}"

import argparse
import sys
import statistics

def main():
    parser = argparse.ArgumentParser(description="Run VLM-as-judge evaluation.")
    parser.add_argument("--result", type=str, required=True, help="Path to the result JSONL file")
    parser.add_argument("--runs", type=int, default=1, help="Total number of evaluation runs per task to calculate mean/std")
    args = parser.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        return

    # Determine final output file path
    suffix = "_vlm_eval.csv" if args.runs == 1 else f"_vlm_eval_{args.runs}runs.csv"
    output_csv = result_path.parent / f"{result_path.stem}{suffix}"
    
    if output_csv.exists():
        print(f"Evaluation already exists: {output_csv}. Skipping.")
        return

    if not BENCHMARK_FILE.exists():
        print(f"Benchmark file not found: {BENCHMARK_FILE}")
        return

    # Check if there's a base evaluation file to resume from
    base_eval_csv = result_path.parent / f"{result_path.stem}_vlm_eval.csv"
    existing_results = {}
    if args.runs > 1 and base_eval_csv.exists() and not output_csv.exists():
        print(f"Found existing base evaluation: {base_eval_csv}. Loading previous scores...")
        try:
            df_existing = pd.read_csv(base_eval_csv)
            for _, row in df_existing.iterrows():
                tid = str(row['task_id'])
                if 'scores' in row and pd.notna(row['scores']):
                    existing_results[tid] = {"scores": json.loads(row['scores']), "reasons": json.loads(row['reasons'])}
                elif 'score' in row:
                    existing_results[tid] = {"scores": [row['score']], "reasons": [str(row.get('reason', ''))]}
        except Exception as e:
            print(f"Error loading existing evaluation: {e}")

    # Load benchmark data
    benchmark_df = pd.read_csv(BENCHMARK_FILE)
    benchmark_data = {}
    for _, row in benchmark_df.iterrows():
        benchmark_data[str(row['ID'])] = {
            "result": row['Result'],
            "task": str(row.get('Task Description', '')),
            "style": str(row.get('Drawing Style', ''))
        }

    # Load executed tasks from result file
    # Deduplicate: Keep only the latest 'success' record, or the last record if no success found.
    task_records = {} # {task_id: [records]}
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                task_result = json.loads(line)
                tid = str(task_result.get('task_id'))
                if tid not in task_records:
                    task_records[tid] = []
                task_records[tid].append(task_result)
            except:
                continue

    executed_tasks = []
    
    # Sort task IDs numerically if possible for cleaner output
    sorted_tids = sorted(task_records.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for tid in sorted_tids:
        records = task_records[tid]
        # Try to find a success record
        success_records = [r for r in records if r.get('status') == 'success']
        if success_records:
            # Use the last success record
            executed_tasks.append(tid)
        else:
            # If no success, use the last record (failed)
            executed_tasks.append(tid)
    
    results = []

    print(f"Evaluating results from: {result_path}")
    
    # Determine the output directory
    # Logic: Since result files are in 'results/...', output files are moved to 'output_results/...'
    # We replace 'results' with 'output_results' in the path to find the output directory.
    
    current_run_output_dir = None
    
    try:
        # Resolve to absolute path to handle relative paths correctly
        abs_result_path = result_path.resolve()
        parts = abs_result_path.parts
        
        if "results" in parts:
            # Find the index of 'results' (use the last occurrence if multiple)
            # e.g. D:/.../results/gpt-4o/...
            idx = len(parts) - 1 - parts[::-1].index("results")
            
            # Construct new path parts: replace 'results' at that index with 'output_results'
            # And append 'output' at the end (as it's inside the agent folder)
            
            # Base path up to 'results'
            new_parts = list(parts[:idx]) + ["output_results"] + list(parts[idx+1:-1]) + ["output"]
            current_run_output_dir = Path(*new_parts)
            
        else:
            # Fallback for non-standard paths (e.g. running from temp dir)
            # Assume output is next to the file
            print(f"Warning: 'results' not found in path {result_path}. Assuming output is in same directory.")
            current_run_output_dir = result_path.parent / "output"
            
    except Exception as e:
        print(f"Error determining output directory: {e}")
        # Fallback
        current_run_output_dir = result_path.parent / "output"

    if not current_run_output_dir.exists():
        # Raise error if local output dir not found
        raise FileNotFoundError(f"Local output directory not found at: {current_run_output_dir}\nPlease ensure 'output' folder is moved correctly to 'output_results'.")
    
    print(f"Using local output dir: {current_run_output_dir}")

    for task_id in executed_tasks:
        if task_id not in benchmark_data:
            print(f"Skipping Task {task_id}: Not found in benchmark")
            continue
            
        task_info = benchmark_data[task_id]
        expected_result = task_info["result"]
        
        if pd.isna(expected_result):
            continue
            
        print(f"Evaluating Task {task_id}...")
        
        scores = []
        reasons = []
        if task_id in existing_results:
            scores.extend(existing_results[task_id]["scores"])
            reasons.extend(existing_results[task_id]["reasons"])
            
        runs_needed = max(0, args.runs - len(scores))
        
        if runs_needed > 0 and len(scores) > 0:
            print(f"  Task already has {len(scores)} run(s). Doing {runs_needed} more run(s)...")
        
        for run_idx in range(runs_needed):
            if str(expected_result).startswith("CHECK:"):
                # Use current_run_output_dir for value metrics
                score, reason = evaluate_value_metric(expected_result, current_run_output_dir)
                # Normalize to 0-100 for consistency if it's 0.0/1.0
                score = score * 100 
            else:
                # Assume image file
                gt_file = DATASET_RESULT_DIR / expected_result
                # Use current_run_output_dir for finding prediction files
                pred_file = find_file(expected_result, current_run_output_dir)
                
                if not gt_file.exists():
                    reason = f"GT file missing: {gt_file}"
                    score = 0
                elif not pred_file:
                    reason = f"Prediction file missing: {expected_result}"
                    score = 0
                else:
                    score, reason = evaluate_image(
                        gt_file, 
                        pred_file, 
                        task_instruction=task_info["task"],
                        style_instruction=task_info["style"]
                    )
            
            scores.append(score)
            reasons.append(reason)
            
            # Deterministic tasks or missing files don't need multiple runs
            if str(expected_result).startswith("CHECK:") or (score == 0 and "missing" in reason):
                remaining = runs_needed - 1 - run_idx
                if remaining > 0:
                    scores.extend([score] * remaining)
                    reasons.extend([reason] * remaining)
                break
        
        # Ensure scores list has exactly args.runs elements (handling any edge cases)
        if len(scores) < args.runs:
            last_s = scores[-1] if scores else 0.0
            last_r = reasons[-1] if reasons else "Padding"
            padding_count = args.runs - len(scores)
            scores.extend([last_s] * padding_count)
            reasons.extend([last_r] * padding_count)
            
        mean_score = sum(scores) / len(scores) if scores else 0.0
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        
        # Round to 2 decimal places
        mean_score = round(mean_score, 2)
        std_score = round(std_score, 2)
        
        print(f"  Mean Score: {mean_score} ± {std_score}")
        results.append({
            "task_id": task_id,
            "score": mean_score,
            "score_std": std_score,
            "scores": json.dumps(scores),
            "reasons": json.dumps(reasons)
        })

    # Calculate Overall Statistics across multiple runs
    # We need to calculate the average score for each run (e.g., Run 1 avg, Run 2 avg, Run 3 avg)
    # Then compute the mean and std of those 3 values.
    
    if results and args.runs > 0:
        run_averages = []
        
        # Iterate through each run index (0 to args.runs-1)
        for i in range(args.runs):
            run_scores = []
            for r in results:
                try:
                    # Parse the scores list for this task
                    task_scores = json.loads(r['scores'])
                    # Get the score for the i-th run, default to 0 if missing (should be padded already)
                    score = task_scores[i] if i < len(task_scores) else 0.0
                    run_scores.append(score)
                except:
                    run_scores.append(0.0)
            
            # Calculate average for this specific run across all tasks
            if run_scores:
                run_avg = sum(run_scores) / len(run_scores)
                run_averages.append(run_avg)
                
        # Now calculate statistics on the run averages
        overall_mean = sum(run_averages) / len(run_averages) if run_averages else 0.0
        overall_std = statistics.stdev(run_averages) if len(run_averages) > 1 else 0.0
        
        print(f"\n{'='*40}")
        print(f"OVERALL SUMMARY ({len(results)} tasks, {args.runs} runs)")
        for idx, val in enumerate(run_averages):
            print(f"  Run {idx+1} Average: {val:.2f}")
        print(f"  ----------------------------------------")
        print(f"  Final Mean: {overall_mean:.2f}")
        print(f"  Final Std Dev: {overall_std:.2f}")
        print(f"{'='*40}\n")
        
        # Append summary row
        results.append({
            "task_id": "OVERALL",
            "score": round(overall_mean, 2),
            "score_std": round(overall_std, 2),
            "scores": json.dumps(run_averages), # Save the per-run averages here
            "reasons": json.dumps([f"Run {i+1} Avg: {v:.2f}" for i, v in enumerate(run_averages)])
        })

    # Save results to the same directory as the result file
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Evaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()