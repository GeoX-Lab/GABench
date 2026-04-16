import asyncio
import sys
import csv
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Import from independent module
sys.path.append(str(Path(__file__).parent.parent))
from agents.base import BaseAgent
from agents.react import ReactAgent
from agents.plan_and_react import PlanAgent, SolveReactAgent
from agents.plan_and_solve import PlanAndSolveAgent
from core.mcp_client import get_mcp_clients

# ================= 配置区域 =================
# 结果保存目录
OUTPUT_DIR = Path("results")
# ===========================================

CSV_PATH = Path(__file__).parent.parent / "benchmark" / "benchmark.csv"

def load_tasks_from_csv(file_path):
    """加载所有测试任务"""
    tasks = []
    if not file_path.exists():
        print(f"Warning: CSV file not found at {file_path}")
        return []
        
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("ID") or row.get("id") or row.get("任务ID")
                desc = (row.get("Task Description") or row.get("任务描述") or "").strip()
                inp_data = (row.get("Data Description") or row.get("输入数据") or "").strip()
                draw_style = (row.get("Drawing Style") or row.get("绘图格式") or "").strip()
                
                if desc:
                    query_parts = [desc]
                    if inp_data:
                        query_parts.append(f"Data: {inp_data}")
                    if draw_style:
                        query_parts.append(f"Drawing Style: {draw_style}")
                        
                    full_query = "\n".join(query_parts)
                    tasks.append({
                        "id": tid,
                        "query": full_query,
                        "meta": row # 保存原始行数据以备查
                    })
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
    return tasks

def get_completed_tasks(result_file):
    """读取已完成（status=success）的任务ID"""
    completed = set()
    if result_file.exists():
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        # 只有成功的任务才跳过，失败的可以重试
                        if data.get("status") == "success":
                            completed.add(str(data.get("task_id")))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading result file {result_file}: {e}")
    return completed

async def run_evaluation(agent_type, model_name, query):
    """
    核心评估函数：运行任务并返回结果字典（不负责保存文件）
    """
    print(f"\n>>> Starting Task [Agent: {agent_type} | Model: {model_name}]")
    print(f">>> Query: {query[:100]}..." if len(query) > 100 else f">>> Query: {query}")

    history = []
    mcp_clients = get_mcp_clients()
    
    start_time = datetime.now()
    status = "success"
    error_msg = None

    try:
        if agent_type == "react":
            agent = ReactAgent(mcp_clients=mcp_clients, init_model_name=model_name)
            async with agent:
                async for chunk in agent.run(query):
                    print(chunk, end="", flush=True)
            history = agent.history

        elif agent_type == "plan_and_react":
            # 1. Plan
            print("\n--- Planning Phase ---")
            planner = PlanAgent(mcp_clients=mcp_clients, init_model_name=model_name)
            async with planner:
                async for chunk in planner.run(query):
                    print(chunk, end="", flush=True)
            history.extend(planner.history)
            
            # 2. Solve
            print("\n--- Solving Phase ---")
            solver = SolveReactAgent(mcp_clients=mcp_clients, init_model_name=model_name)
            
            # Pass the plan to the solver
            solver.set_subtasks(planner.subtasks)
            
            async with solver:
                async for chunk in solver.run(query):
                    print(chunk, end="", flush=True)
            history.extend(solver.history)

        elif agent_type == "plan_and_solve":
            # Combined Plan & Solve Agent
            agent = PlanAndSolveAgent(mcp_clients=mcp_clients, init_model_name=model_name)
            async with agent:
                async for chunk in agent.run(query):
                    print(chunk, end="", flush=True)
            history = agent.history

        elif agent_type == "base":
            # Base Agent (Tool-enabled, no complex architecture)
            agent = BaseAgent(mcp_clients=mcp_clients, init_model_name=model_name)
            async with agent:
                async for chunk in agent.run(query):
                    print(chunk, end="", flush=True)
            history = agent.history

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    except Exception as e:
        print(f"\n!!! Error during evaluation: {e}")
        status = "error"
        error_msg = str(e)

    duration = (datetime.now() - start_time).total_seconds()
    end_time = datetime.now()

    # 按照指定顺序返回结果
    return {
        "agent": agent_type,
        "model": model_name,
        "status": status,
        "error": error_msg,
        "duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "query": query,
        "history": history
    }

async def run_specific_tasks(task_ids_str, agent_type, model_name):
    """运行指定的一个或多个任务（调试模式）- 并在控制台输出，同时保存日志到文件"""
    
    # 解析任务ID列表
    if "," in task_ids_str:
        target_ids = [tid.strip() for tid in task_ids_str.split(",") if tid.strip()]
    else:
        target_ids = [task_ids_str.strip()]
        
    tasks = load_tasks_from_csv(CSV_PATH)
    
    # 准备结果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = OUTPUT_DIR / "debug"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    # 格式: {model}_{agent}_{time}.jsonl
    result_file = save_dir / f"{safe_model_name}_{agent_type}_{timestamp}.jsonl"
    
    print(f"=== Running Specific Tasks (Debug Mode) ===")
    print(f"Target IDs: {target_ids}")
    print(f"Model: {model_name}")
    print(f"Agent: {agent_type}")
    print(f"Log File: {result_file}\n")
    
    for task_id in target_ids:
        task = next((t for t in tasks if str(t["id"]) == str(task_id)), None)
        
        if not task:
            print(f"Error: Task ID {task_id} not found in benchmark CSV. Skipping.")
            continue
        
        print(f"\n>>> Executing Task ID: {task_id}")
        
        result = await run_evaluation(agent_type, model_name, task["query"])
        
        # 构建最终结果字典
        final_result = {
            "task_id": task_id,
            "agent": result["agent"],
            "model": result["model"],
            "status": result["status"],
            "error": result["error"],
            "duration_seconds": result["duration_seconds"],
            "start_time": result.get("start_time"),
            "end_time": result.get("end_time"),
            "query": result["query"],
            "history": result["history"]
        }
        
        try:
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
            print(f"[Result Appended] Task {task_id} -> {result_file}")
        except Exception as e:
            print(f"[Error Saving Log]: {e}")

        # 在控制台显示结果摘要
        print(f"{'-'*30}")
        print(f"Task {task_id} Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']:.2f}s")
        if result.get('error'):
            print(f"Error: {result['error']}")
        print(f"{'-'*30}\n")

async def run_benchmark(model, agent, resume_path=None):
    # 1. 准备
    tasks = load_tasks_from_csv(CSV_PATH)
    if not tasks:
        print("No tasks found in CSV.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if resume_path:
        result_file = Path(resume_path)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== Resuming Benchmark Run from: {result_file} ===")
    else:
        # 简化命名: all_tools_时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"all_tools_{timestamp}"
        print(f"=== Starting New Benchmark Run: {run_id} ===")
    
        # 2. 执行任务
        # 为当前 Model + Agent 组合创建一个结果文件
        save_dir = OUTPUT_DIR / model / agent
        save_dir.mkdir(parents=True, exist_ok=True)
        result_file = save_dir / f"{run_id}.jsonl"

    total_combinations = len(tasks)
    current_count = 0
    
    # 获取已完成的任务列表
    completed_tasks = get_completed_tasks(result_file)
    print(f"--- Running Group: Model={model}, Agent={agent} ---")
    print(f"--- Saving to: {result_file} ---")
    print(f"--- Already completed tasks: {len(completed_tasks)} ---")

    for task in tasks:
        current_count += 1
        task_id = str(task["id"])
        
        if task_id in completed_tasks:
            print(f"[{current_count}/{total_combinations}] ID: {task_id} | Model: {model} | Agent: {agent} -> SKIPPED (Already Success)")
            continue

        query = task["query"]
        
        print(f"\n[{current_count}/{total_combinations}] ID: {task_id} | Model: {model} | Agent: {agent}")
        
        task_start_time = datetime.now()
        try:
            result = await run_evaluation(agent, model, query)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running task {task_id}: {e}")
            task_end_time = datetime.now()
            result = {
                "agent": agent,
                "model": model,
                "status": "error",
                "error": str(e),
                "duration_seconds": (task_end_time - task_start_time).total_seconds(),
                "start_time": task_start_time.isoformat(),
                "end_time": task_end_time.isoformat(),
                "query": query,
                "history": []
            }
        
        # 按照指定顺序构建最终结果
        final_result = {
            "task_id": task_id,
            "agent": result["agent"],
            "model": result["model"],
            "status": result["status"],
            "error": result["error"],
            "duration_seconds": result["duration_seconds"],
            "start_time": result.get("start_time"),
            "end_time": result.get("end_time"),
            "query": result["query"],
            "history": result["history"]
        }
        
        try:
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_result, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            print(f"!!! ERROR writing result to file: {e}")

    print(f"\n=== Benchmark Completed ===")
    
    # 3. 统计失败任务
    print("\n=== Task Status Summary ===")
    
    # 记录所有任务的状态，可能有多次运行记录
    task_status_history = {} # {task_id: [{"status": "success/error", "error": "msg"}]}
    
    # result_file 已经在前面定义并使用了，直接使用即可
    if result_file.exists():
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        task_id = str(data.get("task_id"))
                        status = data.get("status")
                        error_info = data.get("error", "Unknown error")
                        
                        if task_id not in task_status_history:
                            task_status_history[task_id] = []
                        task_status_history[task_id].append({
                            "status": status,
                            "error": error_info
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading result file {result_file}: {e}")
    
    # 筛选出真正失败的任务（即没有一次成功的任务）
    final_failed_tasks = {} # {task_id: error_msg}
    
    for task_id, history in task_status_history.items():
        # 检查是否曾经成功过
        has_success = any(record["status"] == "success" for record in history)
        
        if not has_success:
            # 如果从未成功，取最后一次的错误信息
            last_error = history[-1]["error"]
            final_failed_tasks[task_id] = last_error
            
    if final_failed_tasks:
        print(f"\n❌ Found {len(final_failed_tasks)} failed tasks:")
        print("-" * 80)
        for task_id in sorted(final_failed_tasks.keys()):
            error = final_failed_tasks[task_id]
            print(f"\nTask ID: {task_id}")
            print(f"  - {model}/{agent}: {str(error)[:100]}...")
        print("-" * 80)
        
        # 简洁版本：只列出失败的任务ID
        failed_ids = sorted(final_failed_tasks.keys())
        print(f"\nFailed Task IDs: {', '.join(failed_ids)}")
    else:
        print("\n✅ All tasks completed successfully!")

    # Move generated 'output' folder to output_results
    try:
        source_output_dir = Path("output")
        if source_output_dir.exists() and source_output_dir.is_dir():
            final_output_base = Path(r"d:\study_up\geobench\output_results")
            # Move to output_results/{model}/{agent}/output
            target_dir = final_output_base / model / agent / "output"
            
            # Ensure parent directory exists
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            if target_dir.exists():
                print(f"\n[Info] Target output directory already exists: {target_dir}")
                # Merge contents
                for item in source_output_dir.iterdir():
                    dst_path = target_dir / item.name
                    if dst_path.exists():
                        if dst_path.is_dir():
                            shutil.rmtree(dst_path)
                        else:
                            dst_path.unlink()
                    shutil.move(str(item), str(dst_path))
                # Remove empty source dir
                source_output_dir.rmdir()
                print(f"[Moved] Output contents merged into: {target_dir}")
            else:
                shutil.move(str(source_output_dir), str(target_dir))
                print(f"\n[Moved] Output folder moved to: {target_dir}")
            
    except Exception as e:
        print(f"\n[Error] Failed to move output folder: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoBench Evaluation Runner")
    parser.add_argument("--resume", type=str, help="Resume from a specific result file path (e.g. results/gpt-4o/react/all_tools_xxx.jsonl)")
    parser.add_argument("--id", type=str, help="Run single task by ID (debug mode)")
    parser.add_argument("--agent", type=str, default="react", choices=["react", "plan_and_react", "plan_and_solve", "base"], help="Agent type (default: react)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o)")
    args = parser.parse_args()
    
    # 指定任务模式 (Debug)
    if args.id:
        asyncio.run(run_specific_tasks(args.id, args.agent, args.model))
    # 完整基准测试模式
    else:
        asyncio.run(run_benchmark(model=args.model, agent=args.agent, resume_path=args.resume))