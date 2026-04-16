planner_prompt = """You are an intelligent assistant that needs to solve user problems.

Available tools:
{tools}

User question: {query}

Please analyze the user's question and create an execution plan. The plan is a tool chain that you need to carefully design to make the tool chain can solve the user's question properly. Decide the length of the tool chain based on the difficulty of the problem.

Please return the plan in JSON format as follows:
{{
    "plan": [
        {{
            "tool": "tool_name_1",
            "parameters": {{"param_name": "param_value"}}
        }},
        {{
            "tool": "tool_name_2",
            "parameters": {{"param_name": "param_value"}}
        }}
    ]
}}

Output the JSON strictly within a markdown code block ```json ... ```.
"""

# summary_prompt = """Execute tool calls based on the created plan.

# Available tools:
# {tools}

# User question: {query}

# Execution plan:
# {plan}

# Executed results:
# {results}

# Please summarize the above tool call results and generate a concise answer."""