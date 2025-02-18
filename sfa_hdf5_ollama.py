"""
Single-File HDF5 Agent (SFA-HDF5)

This module implements a natural language interface for exploring HDF5 data files using
a local Language Model (LLM) served by Ollama. The agent provides functionality to:
- List and navigate HDF5 files, groups, and datasets
- Retrieve dataset information, shapes, and data types
- Read and analyze data slices
- Generate dataset summaries

The agent uses a tool-based architecture where the LLM can invoke specific functions
to interact with HDF5 files based on natural language queries.

This is a self-contained single file that uses astral/uv for dependency management.
All dependencies are specified in the script header and will be automatically installed
when the script is run.

Dependencies are managed using astral/uv to ensure reproducibility and portability.
The script can be run directly without any external requirements.txt file.

Author: Anthony Kougkas | https://akougkas.io
License: MIT
"""

# /// script
# package = "sfa-hdf5-agent"
# version = "0.1.0"
# authors = ["Anthony Kougkas | https://akougkas.io"]
# description = "Single-File HDF5 Agent with natural language interface"
# repository = "https://github.com/akougkas/hdf5-agent"
# license = "MIT"
# dependencies = [
#     "h5py>=3.8.0,<3.9.0",
#     "requests~=2.31.0",
#     "rich~=13.3.5",
#     "pydantic>=2.4.2,<3.0.0",
#     "ollama~=0.1.6",
#     "psutil~=5.9.5",
#     "numpy>=1.24.0,<2.0.0"
# ]
# requires-python = ">=3.8,<3.13"
# ///

import h5py
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel, Field
from typing import (
    List, 
    Optional, 
    Tuple, 
    Type, 
    TypeVar, 
    Generic, 
    Dict, 
    Any,
    Union
)
import ollama
import requests
import sys
import os
import re
import json
import time
from functools import lru_cache
import psutil
import argparse

# --- Type Definitions ---
T = TypeVar('T', bound=BaseModel)

# --- Constants ---
MODEL = "phi4:latest"  # Default model to use
OLLAMA_API_TIMEOUT = 30  # Default timeout in seconds
CACHE_SIZE = 128  # Size of LRU cache for metadata
MAX_SLICE_SIZE = 1_000_000  # Maximum number of elements to read in a slice
METADATA_CACHE_TTL = 300  # Time-to-live for metadata cache in seconds
OLLAMA_ENDPOINT = "http://localhost:11434"
MAX_RETRIES = 3
RETRY_DELAY = 1

AGENT_PROMPT = """You are an agent designed to interact with HDF5 files.
Your goal is to help the user explore and understand the contents of HDF5 files.

Working Directory: {{directory_path}}

You must respond in one of two formats:

1. For tool calls:
{
    "tool_call": {
        "tool_name": "<name of tool>",
        "parameters": {
            "reasoning": "<why you are using this tool>",
            ...other parameters specific to the tool
        }
    }
}

2. For final answers:
{
    "response": "<your markdown-formatted response>"
}

Available tools:

1. list_files
   - Lists HDF5 files in a directory
   - Parameters:
     * reasoning: Why you need to list files
     * directory_path: Path to search for HDF5 files (use the working directory path)

2. list_groups
   - Lists groups within an HDF5 file
   - Parameters:
     * reasoning: Why you need to list groups
     * file_path: Path to the HDF5 file

3. list_datasets
   - Lists datasets within an HDF5 group
   - Parameters:
     * reasoning: Why you need to list datasets
     * file_path: Path to the HDF5 file
     * group_path: Path to the group within the file

Instructions:
1. Always respond with valid JSON matching one of the formats above
2. Use tools to gather information before providing final answers
3. Keep reasoning concise but informative
4. Format final answers in Markdown
5. Never attempt to read entire datasets larger than 10MB
6. Always use the provided working directory path for file operations

Now, respond to the following request:
{{user_request}}
"""

# --- Pydantic Models ---
class OllamaResponse(BaseModel):
    """Model for Ollama API responses."""
    model: str
    created_at: str
    message: Dict[str, Any]
    done: bool

class ToolParameters(BaseModel):
    """Base model for tool parameters."""
    reasoning: str = Field(..., description="Reasoning for using this tool")

class ListFilesParameters(ToolParameters):
    directory_path: str = Field(..., description="Path to the directory to list files in")

class ListGroupsParameters(ToolParameters):
    file_path: str = Field(..., description="Path to the HDF5 file")

class ListDatasetsParameters(ToolParameters):
    file_path: str = Field(..., description="Path to the HDF5 file")
    group_path: str = Field(..., description="Path to the HDF5 group")

class ToolCall(BaseModel):
    """Model for tool calls from the LLM."""
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool")

class AgentResponse(BaseModel):
    """Model for agent responses."""
    tool_call: Optional[ToolCall] = None
    response: Optional[str] = None

# Performance metrics storage
METRICS: Dict[str, List[float]] = {
    'api_latency': [],
    'file_operations': [],
    'total_memory': []
}

# Global console and progress
console = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console
)

# Initialize Ollama client with connection check
client = None
try:
    client = ollama.Client(host=OLLAMA_ENDPOINT)
    # Test connection with a simple query
    client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        stream=False
    )
except Exception as e:
    console.print(f"[yellow]Note: Ollama connection not available ({str(e)})[/yellow]")
    console.print("[yellow]Some features may be limited.[/yellow]")

def check_ollama_connection() -> bool:
    """Check if Ollama server is running and accessible.
    
    Returns:
        bool: True if Ollama server is running and accessible, False otherwise
    """
    try:
        if not client:
            return False
            
        # Try to list models to verify connection
        client.list()
        return True
    except Exception as e:
        console.print(f"[yellow]Ollama connection error: {e}[/yellow]")
        return False

def ensure_model_loaded() -> bool:
    """Ensure the required model is loaded in Ollama.
    
    Returns:
        bool: True if model is loaded and ready, False otherwise
    """
    try:
        if not client:
            return False
            
        # Extract model name without version tag
        model_name = MODEL.split(':')[0]
        
        # List available models and check if our model exists
        models = client.list()
        # Models are returned as a list of dicts with 'name' key
        model_exists = any(model.get('name') == model_name for model in models['models'])
        
        if not model_exists:
            console.print(f"[yellow]Model {MODEL} not found. Pulling...[/yellow]")
            client.pull(MODEL)
            console.print(f"[green]Successfully pulled {MODEL}[/green]")
        
        return True
    except Exception as e:
        console.print(f"[red]Error checking/pulling model: {e}[/red]")
        return False

def list_groups(file_path: str) -> str:
    """List all groups in an HDF5 file."""
    try:
        if not os.path.exists(file_path):
            return f"[red]Error: File {file_path} not found[/red]"
            
        with h5py.File(file_path, 'r') as f:
            groups = []
            
            def visitor(name, obj):
                if isinstance(obj, h5py.Group):
                    groups.append(name)
                    
            f.visititems(visitor)
            
            if not groups:
                return "No groups found in the file."
                
            # Format the output with better structure
            output = f"[bold cyan]Groups in {os.path.basename(file_path)}:[/bold cyan]\n"
            
            # Sort groups by depth first, then alphabetically
            sorted_groups = sorted(groups, key=lambda x: (x.count('/'), x))
            
            # Format groups with indentation based on depth
            for group in sorted_groups:
                depth = group.count('/')
                indent = "  " * depth
                group_name = group.split('/')[-1]
                output += f"{indent}[green]•[/green] [cyan]{group_name}[/cyan]\n"
                
            return output
            
    except Exception as e:
        return f"[red]Error listing groups: {str(e)}[/red]"

def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Optional[str]:
    """Execute a tool with the given parameters."""
    try:
        if tool_name == "list_files":
            params = ListFilesParameters(**parameters)
            # List HDF5 files in the directory
            if not os.path.exists(params.directory_path):
                return f"[red]Error: Directory {params.directory_path} not found[/red]"
            
            h5_files = []
            for file in os.listdir(params.directory_path):
                if file.endswith(('.h5', '.hdf5')):
                    file_path = os.path.join(params.directory_path, file)
                    try:
                        with h5py.File(file_path, 'r') as f:
                            size = os.path.getsize(file_path)
                            h5_files.append((file, size))
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not open {file}: {str(e)}[/yellow]")
            
            if not h5_files:
                return "[yellow]No HDF5 files found in the directory.[/yellow]"
            
            # Format output
            output = f"[bold cyan]HDF5 files in {params.directory_path}:[/bold cyan]\n"
            for file, size in sorted(h5_files):
                size_str = f"{size/1024/1024:.2f}MB" if size > 1024*1024 else f"{size/1024:.2f}KB"
                output += f"[green]•[/green] [cyan]{file}[/cyan] ({size_str})\n"
            return output
            
        elif tool_name == "list_groups":
            params = ListGroupsParameters(**parameters)
            return list_groups(params.file_path)
        elif tool_name == "list_datasets":
            params = ListDatasetsParameters(**parameters)
            # Execute list_datasets logic
            pass
        else:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]Error executing tool {tool_name}: {e}[/red]")
        return None

def call_ollama(
    prompt_messages: List[Dict[str, str]],
    system_prompt: str = "",
    response_format: Optional[Type[T]] = None,
    timeout: int = OLLAMA_API_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """Call Ollama API with retries and error handling."""
    if not client:
        console.print("[red]Ollama client not initialized[/red]")
        return None
        
    start_time = time.time()
    task = progress.add_task("[cyan]Calling Ollama API...", total=None)
    
    try:
        for attempt in range(MAX_RETRIES):
            try:
                # Debug print
                console.print("[cyan]Sending request to Ollama...[/cyan]")
                
                response = client.chat(
                    model=MODEL,
                    messages=prompt_messages,
                    stream=False,
                    options={
                        "temperature": 0,
                        "timeout": timeout
                    }
                )
                
                # Debug print
                console.print("[cyan]Received response from Ollama[/cyan]")
                console.print(f"[dim]Debug: {response}[/dim]")
                
                return response
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    progress.update(task, description=f"[yellow]Retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise e
                    
    except Exception as e:
        console.print(f"[red]API call failed: {str(e)}[/red]")
        return None
    finally:
        progress.remove_task(task)

def process_query(directory_path: str, query: str) -> None:
    """Process a user query about HDF5 files."""
    # Create a new progress instance for each query
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as query_progress:
        task = query_progress.add_task("[cyan]Processing query...", total=None)
        
        try:
            # Initialize conversation with the query
            initial_prompt = AGENT_PROMPT.replace("{{user_request}}", query).replace("{{directory_path}}", directory_path)
            prompt_messages = [{"role": "user", "content": initial_prompt}]

            while True:  # Main loop for interaction
                # Process the query through Ollama
                response = call_ollama(prompt_messages)

                if response and 'message' in response:
                    content = response['message']['content']
                    console.print(f"\n[cyan]Processing response...[/cyan]")

                    # Extract JSON from markdown code block if present
                    if '```json' in content:
                        content = content.split('```json\n')[1].split('\n```')[0]

                    try:
                        # Parse the JSON response
                        parsed = json.loads(content)
                        if 'tool_call' in parsed:
                            if 'response' in parsed:
                                console.print("[yellow]Warning: Response contains both tool_call and response. Prioritizing tool_call.[/yellow]")
                            tool_call = parsed['tool_call']
                            console.print(f"[cyan]Executing tool: {tool_call['tool_name']}[/cyan]")

                            # Set directory path if not specified
                            if tool_call['tool_name'] == 'list_files' and (
                                    'directory_path' not in tool_call['parameters'] or
                                    tool_call['parameters']['directory_path'] == '<specify_directory_path_here>'):
                                tool_call['parameters']['directory_path'] = directory_path

                            # Adjust file path to include directory for other tools
                            if 'file_path' in tool_call['parameters']:
                                if not os.path.isabs(tool_call['parameters']['file_path']):
                                    if not tool_call['parameters']['file_path'].startswith(directory_path): # Add this check
                                        tool_call['parameters']['file_path'] = os.path.join(
                                            directory_path,
                                            tool_call['parameters']['file_path']
                                        )

                            result = execute_tool(tool_call['tool_name'], tool_call['parameters'])
                            if result:
                                console.print(result)

                            # Append tool output to prompt messages for next iteration
                            prompt_messages.append({"role": "assistant", "content": content})  # original response
                            prompt_messages.append({"role": "user", "content": f"Tool output:\n```\n{result}\n```"})


                        elif 'response' in parsed:
                            console.print(parsed['response'])
                            break  # Exit loop when final answer is given

                        else:
                            console.print(content)
                            break  # Exit if unexpected response

                    except json.JSONDecodeError as e:
                        console.print(f"[red]Error parsing response: {str(e)}[/red]")
                        console.print(content)
                        break  # Exit on parsing error

                else:
                    console.print("[yellow]No response from model[/yellow]")
                    break  # Exit if no response

        except Exception as e:  # This except block was moved
            console.print(f"[red]Error processing query: {str(e)}[/red]")
            console.print(f"[dim]Debug: {type(e).__name__}: {str(e)}[/dim]")

def get_memory_usage() -> float:
    """Get current memory usage of the process in MB.
    
    Returns:
        float: Current memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_performance_metrics() -> None:
    """Print collected performance metrics."""
    console.print("\n[cyan]Performance Metrics:[/cyan]")
    if METRICS['api_latency']:
        avg_latency = sum(METRICS['api_latency']) / len(METRICS['api_latency'])
        console.print(f"Average API Latency: {avg_latency:.2f}s")
    if METRICS['total_memory']:
        max_memory = max(METRICS['total_memory'])
        console.print(f"Peak Memory Usage: {max_memory:.2f}MB")

def main() -> None:
    """Main function for the HDF5 agent."""
    global MODEL
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='HDF5 Agent with natural language interface')
        parser.add_argument('directory', help='Directory containing HDF5 files')
        parser.add_argument('query', nargs='?', help='Query to process (optional)')
        parser.add_argument('-m', '--model', default=MODEL,
                          help=f'Ollama model to use (default: {MODEL})')
        args = parser.parse_args()
        
        # Set the model to use
        MODEL = args.model
        directory_path = args.directory
        
        # Validate directory path
        if not os.path.isdir(directory_path):
            console.print(f"[red]Error: Directory not found: {directory_path}[/red]")
            sys.exit(1)
        
        # Initial setup with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as setup_progress:
            setup_task = setup_progress.add_task("[cyan]Initializing...", total=None)
            
            # Check Ollama connection and model
            if not check_ollama_connection():
                console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
                console.print("Start Ollama with: ollama serve")
                sys.exit(1)
                
            if not ensure_model_loaded():
                console.print(f"[red]Error: Could not load model {MODEL}[/red]")
                sys.exit(1)
        
        # Track initial memory usage
        initial_memory = get_memory_usage()
        METRICS['total_memory'].append(initial_memory)
        
        # If query is provided as command line argument
        if args.query:
            process_query(directory_path, args.query)
            print_performance_metrics()
            return
        
        # Interactive mode
        console.print("\n[cyan]Enter your queries (type 'exit' to quit):[/cyan]")
        while True:
            try:
                # Update memory tracking
                current_memory = get_memory_usage()
                METRICS['total_memory'].append(current_memory)
                
                # Get user input with prompt
                query = input("\nQuery> ").strip()
                if not query:  # Skip empty queries
                    continue
                if query.lower() in ['exit', 'quit']:
                    break
                
                # Process query
                process_query(directory_path, query)
                
                # Cleanup if memory usage is high
                if current_memory > initial_memory * 2:
                    console.print("[yellow]Cleaning up cache...[/yellow]")
                    get_dataset_metadata.cache_clear()
                    get_dataset_info.cache_clear()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation interrupted by user[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                continue
        
        # Print final performance metrics
        print_performance_metrics()
        
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
