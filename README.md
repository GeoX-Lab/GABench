# GeoAgentBench

GeoAgentBench is a benchmark framework for evaluating the capabilities of Large Language Models (LLMs) in geospatial tasks. It includes a series of geospatial analysis tasks, supporting multiple Agent architectures (e.g., ReAct, Plan & Solve) and various models.

GeoAgentBench 是一个用于评估大语言模型（LLM）在地理空间任务中能力的基准测试框架。它包含了一系列地理分析任务，支持多种 Agent 架构（如 ReAct, Plan & Solve）和多种模型。

---

## Table of Contents / 目录

- [English Version](#english-version)
- [中文版本](#中文版本)

---

# English Version

## 🛠️ Environment Setup

This repository uses Git LFS to manage large datasets. Please ensure you have Git LFS installed and run `git lfs install` in your local clone.

### 1. Install Dependencies

#### Option A: Using `uv` (Recommended)

This project recommends using `uv` for environment management and dependency installation. Since the project includes `pyproject.toml` and `uv.lock`, use the standard `uv` workflow:

```bash
# 1. Install uv (if not already installed)
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies (automatically creates virtual environment and installs dependencies)
uv sync
```

#### Option B: Using `requirement.txt`

If you prefer using `pip`, you can install dependencies via `requirement.txt`:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirement.txt
```

### 2. Configuration

The project is configured mainly through `config.yaml` and environment variables.

1.  **LLM Configuration**: Edit the `config.yaml` file and configure your model information (API Key, Base URL, etc.) under the `llm` field.
2.  **Environment Variables**: Create a `.env` file (or set environment variables directly) to store sensitive information such as API Keys.
3.  **MCP Server Configuration**: Configure the MCP server mode under the `mcp_server` field in `config.yaml`.
    *   **stdio (default)**: Automatically starts the MCP server on each run, no extra steps needed.
    *   **http**: Requires manually starting the MCP server.
4.  **VLM Evaluation Configuration**: Configure the VLM judge model for visual evaluation under the `evaluation` field in `config.yaml`. This is used by `evaluation/vlm_judge.py` to score agent-generated maps against ground truth using a vision-language model (e.g., GPT-4o).

`config.yaml` example:
```yaml
llm:
  gpt-4o:
    model: gpt-4o
    base_url: ${BASE_URL}  # Reads from environment variable BASE_URL
    api_key: ${API_KEY}    # Reads from environment variable API_KEY

mcp_server:
  mode: stdio # http or stdio

evaluation:
  judge_model: gpt-4o       # VLM judge model name
  base_url: ${BASE_URL}     # API base URL for VLM evaluation
  api_key: ${API_KEY}       # API key for VLM evaluation
```

### 3. Start MCP Server (HTTP Mode Only)

If `mcp_server.mode` is set to `http` in `config.yaml`, you need to manually start the MCP server before running tests:

```bash
uv run core/mcp_server.py
```

The server will start at `http://127.0.0.1:8000` (configurable in `config.yaml`).

## 🚀 Running Tests

### 1. Run Full Benchmark

Use the `runners/run_benchmark.py` script to run the full benchmark. Specify the model and agent via command-line arguments.

```bash
# Run with specified model and agent (supported: react, plan_and_react, plan_and_solve, base)
uv run runners/run_benchmark.py --model qwen3-32b --agent react
```

**Resume from Checkpoint**:
If a test is interrupted, use the `--resume` parameter with the previous Run ID to continue (model and agent must also be specified):

```bash
uv run runners/run_benchmark.py --model gpt-4o --agent react --resume all_tools_20250101_120000
```

### 2. Run Single or Multiple Tasks (Debug Mode)

Single-task debugging is integrated into `runners/run_benchmark.py`, supporting the `--id` parameter to specify one or more task IDs (comma-separated).

```bash
# Run by task ID (IDs from benchmark/benchmark.csv)
# Run a single task
uv run runners/run_benchmark.py --model gpt-4o --agent react --id 1

# Run multiple tasks
uv run runners/run_benchmark.py --model gpt-4o --agent react --id 1,2,5
```

**Debug Logs**:
Debug mode results are saved in the `results/debug` directory, with filenames like `{model}_{agent}_{timestamp}.jsonl` (containing all task results from that run).

### 3. Replay Tool Logic (Replay Logs)

If you already have a run log file (`.jsonl`) and want to **re-execute only the tool call logic** recorded in the log without going through the LLM (e.g., to reproduce tool errors, regenerate output files, etc.), use the `runners/replay_logs.py` script.

This script parses the `Action` blocks in the log and calls the same tool functions.

```bash
# Replay all tasks from a specified log file
uv run runners/replay_logs.py --log_file results/qwen3-32b/react/all_tools_20250101_120000.jsonl

# Replay only a specific task (e.g., Task 44)
uv run runners/replay_logs.py --log_file results/qwen3-32b/react/all_tools_20250101_120000.jsonl --task_id 44
```

**Note**:
*   This mode completely bypasses the LLM and only executes tools.
*   It uses the current `config.yaml` settings (e.g., `output_dir`), so it can be used to reproduce results in a new output directory.

## ➕ Adding New Models

To test a new model, follow these steps:

1.  **Edit `config.yaml`**: Add the new model configuration under the `llm` section.
    ```yaml
    llm:
      new-model-name:
        model: actual-model-name-on-api
        base_url: http://your-api-endpoint
        api_key: your-api-key
    ```

2.  **Run Tests**: Use the new model name (matching the key in `config.yaml`) in the command line.
    ```bash
    uv run runners/run_benchmark.py --model new-model-name
    ```

## 📊 Viewing Results & Evaluation

### 1. Result Files

*   **Evaluation Logs**: All evaluation results (JSONL format) are saved in `results/<model>/<agent>/<run_id>.jsonl` (e.g., `results/gpt-4o/react/all_tools_20250101_120000.jsonl`).
*   **Tool Outputs**: Intermediate files generated by tools (e.g., layers, statistics) are saved in the `output_dir` configured in `config.yaml` (default `./output`).
    *   **Important**: Since all experiments share the same output directory, **parallel runs are not supported**. It is recommended to **manually back up or clear** this directory before starting a new Benchmark Run to prevent file confusion or overwriting.
*   **Debug Logs**: Debug mode results are output to the console and also saved in the `results/debug` directory.

### 2. Evaluation Metrics

#### 2.1 Step-by-Step Evaluation (Trajectory-Level)

Evaluates the structural coherence of the Agent's tool-invocation sequences and parameter precision.

```bash
# Example: Evaluate results for qwen3-32b model with react Agent
uv run evaluation/step_by_step.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**Metrics**:
*   **TAO (Tools-Any-Order)**: Evaluates whether the model selected the correct tools regardless of order (Recall, Precision, F1).
*   **TIO (Tools-In-Order)**: Evaluates whether the tool call order matches expectations (allowing other tools in between).
*   **TEM (Tools-Exact-Match)**: Evaluates whether the tool call chain exactly matches the ground truth in strict order.
*   **PEA (Parameter Execution Accuracy)**: Evaluates whether tool call parameters are correct, using a "Last-Attempt Alignment" strategy to isolate the final successful invocation from intermediate trial-and-error logs.

#### 2.2 End-to-End Evaluation

Focuses on the quality of the final task output (accuracy and efficiency).

> **Important**: Since all experiments share the same output directory (default `./output`) without version isolation, **please ensure the `output` directory contains the generated files corresponding to the Run ID being evaluated**. If new experiments have been run, old files may have been overwritten, leading to inaccurate evaluation results.

**VLM-as-Judge Evaluation (Vision Model Based)**:

Uses a powerful vision model (e.g., GPT-4o) as a judge to compare and score agent-generated maps against ground truth (GT). Configure the judge model in the `evaluation` section of `config.yaml`.

```bash
# Run VLM scoring (generates _vlm_eval.csv)
uv run evaluation/vlm_judge.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**VLM Metrics**:
*   **VLM Average Score**: The vision model's score for similarity between generated results and GT (0-100).

**Trajectory Execution Efficiency**:

Evaluates workflow redundancy and resource utilization efficacy based on tool-invocation trajectories.

```bash
# Run efficiency evaluation
uv run evaluation/end_to_end.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**Efficiency Metrics**:
*   **Eff_macro (Macro-Average Step Efficiency)**: The average per-task step efficiency across all successfully completed tasks. For each task *i*, Eff(i) = N_gt(i) / max(N_gt(i), N_pred(i)).
*   **Eff_micro (Micro-Average Step Efficiency)**: The global resource utilization rate, computed as the sum of ground-truth steps divided by the sum of max(gt, pred) steps across all successful tasks.
*   **Note**: Efficiency metrics are strictly bounded within [0, 1]. Only successfully completed tasks are included in the calculation.

### 3. Token Usage Statistics

Use `evaluation/analyze_llm_usage.py` to track LLM token consumption (including Input, Output, TTFT, Speed, etc.).

```bash
# Statistics for all models
uv run evaluation/analyze_llm_usage.py

# Statistics for a specific model
uv run evaluation/analyze_llm_usage.py --model Qwen3-32B-AWQ

# [Recommended] Auto-statistics based on experiment result file
uv run evaluation/analyze_llm_usage.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

---

# 中文版本

## 🛠️ 环境准备

本仓库使用 Git LFS 管理大体积数据集。请先确保你已安装 Git LFS，并在本地克隆仓库后执行 `git lfs install`。

### 1. 安装依赖

#### 方式 A：使用 `uv`（推荐）

本项目推荐使用 `uv` 进行环境管理和依赖安装。由于项目中包含 `pyproject.toml` 和 `uv.lock`，请使用标准 `uv` 流程：

```bash
# 1. 安装 uv (如果尚未安装)
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步依赖 (会自动创建虚拟环境并安装依赖)
uv sync
```

#### 方式 B：使用 `requirement.txt`

如果你更习惯使用 `pip`，可以通过 `requirement.txt` 安装依赖：

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirement.txt
```

### 2. 配置环境

项目主要通过 `config.yaml` 和环境变量进行配置。

1.  **LLM 配置**: 修改 `config.yaml` 文件，在 `llm` 字段下配置你的模型信息（API Key, Base URL 等）。
2.  **环境变量**: 创建 `.env` 文件（或直接设置环境变量）来存储敏感信息，如 API Key。
3.  **MCP Server 配置**: 在 `config.yaml` 的 `mcp_server` 字段配置 MCP 服务器的运行模式。
    *   **stdio (默认)**: 每次运行时自动启动 MCP 服务器，无需额外操作。
    *   **http**: 需要手动启动 MCP 服务器。
4.  **VLM 评估配置**: 在 `config.yaml` 的 `evaluation` 字段配置用于视觉评估的 VLM 裁判模型。该配置被 `evaluation/vlm_judge.py` 使用，通过视觉语言模型（如 GPT-4o）对 Agent 生成的地图与标准答案进行对比评分。

`config.yaml` 示例:
```yaml
llm:
  gpt-4o:
    model: gpt-4o
    base_url: ${BASE_URL}  # 会读取环境变量 BASE_URL
    api_key: ${API_KEY}    # 会读取环境变量 API_KEY

mcp_server:
  mode: stdio # http or stdio

evaluation:
  judge_model: gpt-4o       # VLM 裁判模型名称
  base_url: ${BASE_URL}     # VLM 评估用的 API 地址
  api_key: ${API_KEY}       # VLM 评估用的 API Key
```

### 3. 启动 MCP 服务器 (仅 HTTP 模式)

如果 `config.yaml` 中配置 `mcp_server.mode` 为 `http`，则需要在运行测试前手动启动 MCP 服务器：

```bash
uv run core/mcp_server.py
```

服务器将在 `http://127.0.0.1:8000` 启动（可在 `config.yaml` 中配置 host 和 port）。

## 🚀 运行测试

### 1. 运行完整 Benchmark

使用 `runners/run_benchmark.py` 脚本运行完整的基准测试。需通过命令行参数指定模型和 Agent。

```bash
# 运行指定模型和 Agent (支持: react, plan_and_react, plan_and_solve, base)
uv run runners/run_benchmark.py --model qwen3-32b --agent react
```

**断点续测**:
如果测试意外中断，可以使用 `--resume` 参数指定上一次运行的 Run ID 继续运行（需同时指定模型和 Agent）：

```bash
uv run runners/run_benchmark.py --model gpt-4o --agent react --resume all_tools_20250101_120000
```

### 2. 运行单个或多个任务 (调试模式)

单任务调试功能已集成到 `runners/run_benchmark.py` 中，支持通过 `--id` 参数指定一个或多个任务 ID（逗号分隔）。

```bash
# 按任务 ID 运行 (ID 来自 benchmark/benchmark.csv)
# 运行单个任务
uv run runners/run_benchmark.py --model gpt-4o --agent react --id 1

# 运行多个任务
uv run runners/run_benchmark.py --model gpt-4o --agent react --id 1,2,5
```

**调试日志**:
调试模式的运行结果会保存在 `results/debug` 目录下，文件名为 `{model}_{agent}_{timestamp}.jsonl`（包含该次运行的所有任务结果）。

### 3. 重新执行工具逻辑 (Replay Logs)

如果你已经有了一次运行的日志文件（`.jsonl`），并且希望**不经过 LLM**，仅重新执行日志中记录的工具调用逻辑（例如为了复现工具报错、重新生成输出文件等），可以使用 `runners/replay_logs.py` 脚本。

该脚本会解析日志中的 `Action` 块，并调用相同的工具函数。

```bash
# 重新执行指定日志文件中的所有任务
uv run runners/replay_logs.py --log_file results/qwen3-32b/react/all_tools_20250101_120000.jsonl

# 仅重新执行特定任务 (例如 Task 44)
uv run runners/replay_logs.py --log_file results/qwen3-32b/react/all_tools_20250101_120000.jsonl --task_id 44
```

**注意**:
*   此模式完全绕过 LLM，仅执行工具。
*   它使用当前的 `config.yaml` 配置（如 `output_dir`），因此可以用来在新的输出目录中重现结果。

## ➕ 添加新模型

要测试一个新的模型，请按照以下步骤操作：

1.  **修改 `config.yaml`**: 在 `llm` 部分添加新模型的配置。
    ```yaml
    llm:
      new-model-name:
        model: actual-model-name-on-api
        base_url: http://your-api-endpoint
        api_key: your-api-key
    ```

2.  **运行测试**: 直接在命令行中使用新模型的名称（与 `config.yaml` 中的键名一致）。
    ```bash
    uv run runners/run_benchmark.py --model new-model-name
    ```

## 📊 结果查看与评估

### 1. 查看结果文件

*   **评测日志**: 所有的评测结果（JSONL 格式）保存在 `results/<model>/<agent>/<run_id>.jsonl`（例如 `results/gpt-4o/react/all_tools_20250101_120000.jsonl`）。
*   **工具输出**: 工具生成的中间文件（如图层、统计数据）保存在 `config.yaml` 中配置的 `output_dir` (默认 `./output`)。
    *   **重要提示**: 由于所有实验共享同一个输出目录，**不支持**并行运行多个实验。建议在每次启动新的 Benchmark Run 之前，**手动备份或清空**该目录，以防止结果文件混淆或覆盖。
*   **调试日志**: 调试模式运行的结果会直接在控制台输出，并同时保存在 `results/debug` 目录。

### 2. 计算评估指标 (Evaluation)

#### 2.1 逐步评估 (Trajectory-Level Evaluation)

评估 Agent 工具调用序列的结构一致性和参数精确度。

```bash
# 示例：评估 qwen3-32b 模型 react Agent 的结果
uv run evaluation/step_by_step.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**指标说明**:
*   **TAO (Tools-Any-Order)**: 评估模型是否选择了正确的工具，不考虑顺序（Recall, Precision, F1）。
*   **TIO (Tools-In-Order)**: 评估模型调用的工具顺序是否符合预期（允许中间插入其他工具）。
*   **TEM (Tools-Exact-Match)**: 评估工具调用链是否与标准答案在严格顺序上完全一致。
*   **PEA (Parameter Execution Accuracy)**: 评估工具调用的参数是否正确，采用「Last-Attempt Alignment」策略，从中间试错日志中隔离最终成功调用进行评估。

#### 2.2 端到端评估 (End-to-End Evaluation)

关注任务最终产出的结果质量（准确性与效率）。

> **重要提示**: 由于所有实验共享同一个输出目录 (默认 `./output`)，且不支持版本隔离，**评估时请务必确认 `output` 目录中保留的是当前要评估的 Run ID 对应的生成文件**。如果运行了新的实验，旧的文件可能已被覆盖，导致评估结果不准确。

**VLM-as-Judge 评估 (基于视觉模型)**:

使用强大的视觉模型（如 GPT-4o）作为裁判，对 Agent 生成的图表与标准答案（GT）进行对比评分。裁判模型在 `config.yaml` 的 `evaluation` 部分配置。

```bash
# 运行 VLM 评分 (生成 _vlm_eval.csv)
uv run evaluation/vlm_judge.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**VLM 指标说明**:
*   **VLM 平均得分 (Average Score)**: 视觉模型对生成结果与 GT 相似度的打分（0-100）。

**轨迹执行效率 (Trajectory Execution Efficiency)**:

评估工作流冗余度和资源利用效率，基于工具调用轨迹进行量化。

```bash
# 运行效率评估
uv run evaluation/end_to_end.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```

**效率指标说明**:
*   **Eff_macro (宏平均步骤效率)**: 所有成功完成任务的逐任务步骤效率的算术平均。对于第 *i* 个任务，Eff(i) = N_gt(i) / max(N_gt(i), N_pred(i))。
*   **Eff_micro (微平均步骤效率)**: 全局资源利用率，计算为所有成功任务的标准步数之和除以 max(gt, pred) 步数之和。
*   **注意**: 效率指标严格限制在 [0, 1] 区间内，仅统计成功完成的任务。

### 3. Token 消耗统计

使用 `evaluation/analyze_llm_usage.py` 统计 LLM 的 Token 消耗情况（包括 Input, Output, TTFT, Speed 等）。

```bash
# 统计所有模型的消耗
uv run evaluation/analyze_llm_usage.py

# 统计特定模型的消耗
uv run evaluation/analyze_llm_usage.py --model Qwen3-32B-AWQ

# [推荐] 根据实验结果文件自动统计消耗
uv run evaluation/analyze_llm_usage.py --result results/qwen3-32b/react/all_tools_20250101_120000.jsonl
```
