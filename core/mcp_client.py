import os
import sys
import yaml
import logging
import httpx
import asyncio
from typing import List, Tuple, Any
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

with open("config.yaml", "r") as f:
    raw_config = os.path.expandvars(f.read())
    config = yaml.safe_load(raw_config)

class StdioClientAdapter:
    """
    Adapter to make mcp stdio client compatible with fastmcp.Client usage in this project.
    """
    def __init__(self, command: str, args: List[str], timeout: float = 1200.0):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        self.session = None
        self._stdio_ctx = None
        self._session_ctx = None
        self.timeout = timeout

    async def _start_client(self):
        """Helper to start the client"""
        self._stdio_ctx = stdio_client(self.server_params)
        read, write = await self._stdio_ctx.__aenter__()
        
        self._session_ctx = ClientSession(read, write)
        self.session = await self._session_ctx.__aenter__()
        
        await self.session.initialize()

    async def _stop_client(self):
        """Helper to stop the client"""
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
            self._session_ctx = None
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
        self.session = None

    async def restart(self):
        """Restart the client process"""
        logging.warning("[Client] Restarting Stdio Client due to timeout/error...")
        await self._stop_client()
        await self._start_client()
        logging.info("[Client] Stdio Client restarted successfully.")

    async def __aenter__(self):
        await self._start_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stop_client()

    async def list_tools(self):
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        try:
            result = await asyncio.wait_for(
                self.session.list_tools(),
                timeout=60.0 # Short timeout for listing tools
            )
            return result.tools
        except asyncio.TimeoutError:
            logging.error(f"[Client] list_tools timed out after 60 seconds")
            await self.restart()
            raise

    async def call_tool(self, name: str, arguments: dict):
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(name, arguments),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            msg = f"Tool '{name}' timed out after {self.timeout} seconds. The tool execution was terminated."
            logging.error(f"[Client] {msg}")
            
            # Restart the client to clear the blocked state
            await self.restart()
            
            # Return error as result so model sees it
            class ErrorResult:
                def __init__(self, msg):
                    self.data = f"Error: {msg}"
            
            return ErrorResult(msg)

        class ResultWrapper:
            def __init__(self, content):
                # Extract text from content items and join them
                text_parts = []
                for item in content:
                    if hasattr(item, "text"):
                        text_parts.append(item.text)
                    else:
                        text_parts.append(str(item))
                self.data = "\n".join(text_parts)
        return ResultWrapper(result.content)

class HttpClientWrapper:
    """
    Wrapper for fastmcp.Client to handle automatic reconnection.
    """
    def __init__(self, url: str, timeout: float = 1200.0):
        self.url = url
        self.timeout = timeout
        # Initialize Client without arguments as it doesn't support custom httpx_client
        self.client = Client(url)
        
        # Patch the underlying httpx client to increase timeout
        # fastmcp default timeout is too short for some geospatial operations
        try:
            timeout_config = httpx.Timeout(self.timeout, connect=60.0)
            patched = False
            # Check common internal attribute names for the httpx client
            for attr in ["_client", "_http_client", "httpx_client", "client"]:
                if hasattr(self.client, attr):
                    val = getattr(self.client, attr)
                    if isinstance(val, httpx.AsyncClient):
                        val.timeout = timeout_config
                        patched = True
                        break
            if not patched:
                logging.warning("【WARNING】Could not find underlying httpx client to patch timeout.")
        except Exception as e:
            logging.warning(f"【WARNING】Failed to patch fastmcp client timeout: {e}")

    async def __aenter__(self):
        return await self.client.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def list_tools(self):
        try:
            return await self.client.list_tools()
        except Exception as e:
            if "Client is not connected" in str(e):
                logging.warning(f"【WARNING】MCP Client disconnected from {self.url} during list_tools. Retrying with temporary connection...")
                async with self.client:
                    return await self.client.list_tools()
            raise e

    async def call_tool(self, name: str, arguments: dict):
        try:
            result = await self.client.call_tool(name, arguments)
            
            # Wrap result to ensure consistent .data attribute with string content
            class ResultWrapper:
                def __init__(self, content):
                    text_parts = []
                    for item in content:
                        if hasattr(item, "text"):
                            text_parts.append(item.text)
                        else:
                            text_parts.append(str(item))
                    self.data = "\n".join(text_parts)
            
            return ResultWrapper(result.content)
            
        except Exception as e:
            if "Client is not connected" in str(e):
                logging.warning(f"【WARNING】MCP Client disconnected from {self.url} during call_tool({name}). Retrying with temporary connection...")
                async with self.client:
                    result = await self.client.call_tool(name, arguments)
                    
                    class ResultWrapper:
                        def __init__(self, content):
                            text_parts = []
                            for item in content:
                                if hasattr(item, "text"):
                                    text_parts.append(item.text)
                                else:
                                    text_parts.append(str(item))
                            self.data = "\n".join(text_parts)
                    return ResultWrapper(result.content)
            raise e

def get_mcp_clients() -> List[Tuple[str, Any]]:
    """
    Initialize and return a list of MCP clients based on the configuration.
    """
    mcp_clients = []
    
    # Use config
    mcp_config = config.get("mcp_server", {})
    mode = mcp_config.get("mode", "http")
    timeout = mcp_config.get("timeout", 1200.0) # Default to 1200s if not set
    
    if mode == "http":
        http_config = mcp_config.get("http", {})
        host = http_config.get("host", "127.0.0.1")
        port = http_config.get("port", 8000)
        url = f"http://{host}:{port}/mcp"
        mcp_clients.append((url, HttpClientWrapper(url, timeout=float(timeout))))
    elif mode == "stdio":
        stdio_config = mcp_config.get("stdio", {})
        executable = stdio_config.get("command", "python")
    
        if executable == "python":
            executable = sys.executable
            
        args_config = stdio_config.get("args", "")
        
        # Handle args whether it's a string or list
        if isinstance(args_config, list):
            extra_args = [str(a) for a in args_config]
        else:
            extra_args = str(args_config).split()
            
        args = extra_args
        # Use our adapter
        mcp_clients.append(("stdio", StdioClientAdapter(command=executable, args=args, timeout=float(timeout))))
        
    return mcp_clients