import sys
import inspect
import importlib
import yaml
import os
from urllib.parse import urlparse
from pathlib import Path
from fastmcp import FastMCP

# Add current directory to sys.path to ensure we can import toolbox modules
current_dir = Path(__file__).parent
if (current_dir / "toolbox").exists():
    root_dir = current_dir
else:
    root_dir = current_dir.parent
    
sys.path.append(str(root_dir))

# Create the MCP server
mcp = FastMCP("GeoBench Toolbox")

# Exclude tools.py and __init__.py to avoid duplicates or issues
EXCLUDED_FILES = {"tools.py", "__init__.py"}

toolbox_dir = root_dir / "toolbox"

if not toolbox_dir.exists():
    sys.stderr.write(f"Error: Toolbox directory not found at {toolbox_dir}\n")
else:
    # Iterate over all .py files in the toolbox directory
    for py_file in toolbox_dir.glob("*.py"):
        if py_file.name in EXCLUDED_FILES:
            continue
            
        module_name = py_file.stem
        full_module_name = f"toolbox.{module_name}"
        
        try:
            # Import the module
            module = importlib.import_module(full_module_name)
            
            # Find all functions defined in the module
            for func_name, func in inspect.getmembers(module, inspect.isfunction):
                # Check if the function is defined in this module (not imported)
                # And ignore private functions (starting with _)
                if func.__module__ == full_module_name and not func_name.startswith("_"):
                    try:
                        # Use the decorator syntax programmatically
                        mcp.tool()(func)
                    except Exception as e:
                        sys.stderr.write(f"Failed to register tool {func_name} from {module_name}: {e}\n")
                        
        except Exception as e:
            sys.stderr.write(f"Error importing module {module_name}: {e}\n")

if __name__ == "__main__":
    # Load config directly
    config_path = root_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            raw_config = os.path.expandvars(f.read())
            config = yaml.safe_load(raw_config)
    else:
        config = {}
        
    mcp_config = config.get("mcp_server", {})
    mode = mcp_config.get("mode", "http")
    
    if mode == "http":
        http_config = mcp_config.get("http", {})
        host = http_config.get("host", "127.0.0.1")
        port = http_config.get("port", 8000)
            
        # Start an HTTP server
        mcp.run(transport="http", host=host, port=port)
    else:
        # Run in stdio mode
        mcp.run()