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
#     "rich~=13.3.5",
#     "pydantic>=2.4.2,<3.0.0",
#     "ollama~=0.1.6",
#     "psutil~=5.9.5",
#     "numpy>=1.24.0,<2.0.0"
# ]
# requires-python = ">=3.8,<3.13"
# ///

import h5py
from typing import Optional, List, Dict, Any, Type, TypeVar
import asyncio
from rich.table import Table
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
import sys
import os
import json
import time
import psutil
import argparse

# --- Type Definitions ---
T = TypeVar('T', bound=BaseModel)

# --- Constants ---
OLLAMA_ENDPOINT = 'http://localhost:11434'
MODEL = "granite3.1-dense:latest"  # Default model to use
OLLAMA_API_TIMEOUT = 30  # Default timeout in seconds
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


# Global console and progress
console = Console()
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console
)

async def initialize_ollama_client() -> Optional[ollama.AsyncClient]:
    """Initialize and test connection to Ollama API.
    
    Returns:
        Optional[ollama.AsyncClient]: Client instance if successful, None otherwise.
    """
    try:
        client = None
        model_names = []
        model_table = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=False,
            refresh_per_second=1,
        ) as task_progress:
            # Step 1: Connect to Ollama
            connect_task = task_progress.add_task("[cyan]Connecting to Ollama...[/cyan]", total=1)
            client = ollama.AsyncClient(host=OLLAMA_ENDPOINT)
            task_progress.update(connect_task, advance=1, description="[green]✓ Connected to Ollama[/green]")
            task_progress.refresh()

            # Step 2: List available models
            list_task = task_progress.add_task("[cyan]Listing available models...[/cyan]", total=1)
            try:
                models = await client.list()
                
                # Handle ListResponse object which has a models attribute containing Model objects
                if hasattr(models, 'models'):
                    model_names = sorted([model.model for model in models.models])
                else:
                    # Fallback for any other response format
                    model_names = []
                
                if not model_names:
                    task_progress.update(list_task, advance=1, description="[yellow]! No models found[/yellow]")
                else:
                    task_progress.update(list_task, advance=1, description=f"[green]✓ Found {len(model_names)} models[/green]")
                    model_table = Table(show_header=False, box=None, padding=(0, 2))
                    for name in model_names:
                        model_table.add_row(f"[cyan]•[/cyan]", name)
            except Exception as e:
                task_progress.update(list_task, advance=1, description=f"[red]✗ Failed to list models: {str(e)}[/red]")
                model_names = []
            task_progress.refresh()

        # Print available models after progress is complete
        if model_names:
            console.print("\n[bold cyan]Available Models[/bold cyan]")
            console.print(model_table)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=False,
            refresh_per_second=1,
        ) as task_progress:
            # Step 3: Check and pull model if needed
            model_task = task_progress.add_task(f"[cyan]Checking model {MODEL}...[/cyan]", total=1)
            if MODEL in model_names:
                task_progress.update(model_task, advance=1, description=f"[green]✓ Default model '{MODEL}' already available[/green]")
            else:
                task_progress.update(model_task, description=f"[cyan]Model {MODEL} not found - pulling from Ollama...[/cyan]")
                try:
                    await client.pull(model=MODEL)
                    task_progress.update(model_task, advance=1, description=f"[green]✓ Model {MODEL} pulled successfully[/green]")
                except Exception as e:
                    task_progress.update(model_task, advance=1, description=f"[red]✗ Failed to pull model {MODEL}: {str(e)}[/red]")
                    raise
            task_progress.refresh()

            # Step 4: Test model with a simple query
            test_task = task_progress.add_task("[cyan]Testing model...[/cyan]", total=1)
            try:
                test_response = await client.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Reply with 'OK' only"}],
                    stream=False
                )
                # The response format varies between Ollama versions, handle both cases
                if hasattr(test_response, 'message'):
                    response_text = test_response.message.get('content', '')
                elif isinstance(test_response, dict):
                    response_text = test_response.get('response', '')
                else:
                    raise ValueError(f"Unexpected response format: {type(test_response)}")
                
                task_progress.update(test_task, advance=1, description="[green]✓ Model test successful[/green]")
                task_progress.refresh()
            except Exception as e:
                task_progress.update(test_task, advance=1, description=f"[red]✗ Model test failed: {str(e)}[/red]")
                raise ValueError(f"Model test failed: {str(e)}")

        console.print(f"\n[green]Using model:[/green] [cyan]{MODEL}[/cyan]\n")
        return client

    except Exception as e:
        console.print(f"[red]Error initializing Ollama client: {str(e)}[/red]")
        return None

# Initialize the Ollama client
client = asyncio.run(initialize_ollama_client())

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
            try:
                if not os.path.exists(params.file_path):
                    return f"[red]Error: File {params.file_path} not found[/red]"

                with h5py.File(params.file_path, 'r') as f:
                    if params.group_path not in f:
                        return f"[red]Error: Group {params.group_path} not found in {params.file_path}[/red]"

                    datasets = []
                    def visitor(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            datasets.append(name)

                    f[params.group_path].visititems(visitor)

                    if not datasets:
                        return "No datasets found in the specified group."

                    # Format output
                    output = f"[bold cyan]Datasets in {params.group_path} of {os.path.basename(params.file_path)}:[/bold cyan]\n"
                    for dataset in datasets:
                        # Get relative path for display
                        relative_path = dataset.replace(params.group_path, '').lstrip('/')
                        output += f"[green]•[/green] [cyan]{relative_path}[/cyan]\n"
                    return output

            except Exception as e:
                return f"[red]Error listing datasets: {str(e)}[/red]"
        else:
            console.print(f"[red]Unknown tool: {tool_name}[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]Error executing tool {tool_name}: {e}[/red]")
        return None

async def call_ollama(
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
                
                response = await client.chat(
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

async def process_query(directory_path: str, query: str) -> None:
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
                response = await call_ollama(prompt_messages)

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
        # Initialize Ollama client
        global client
        client = asyncio.run(initialize_ollama_client())
        if client is None:
            console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
            console.print("Start Ollama with: ollama serve")
            sys.exit(1)
        
        # Track initial memory usage
        initial_memory = get_memory_usage()
        
        # If query is provided as command line argument
        if args.query:
            asyncio.run(process_query(directory_path, args.query))
            return
        
        # Interactive mode
        console.print("\n[cyan]Enter your queries (type 'exit' to quit):[/cyan]")
        while True:
            try:
                # Update memory tracking
                current_memory = get_memory_usage()
                
                # Get user input with prompt
                query = input("\nQuery> ").strip()
                if not query:  # Skip empty queries
                    continue
                if query.lower() in ['exit', 'quit']:
                    break
                
                # Process query
                asyncio.run(process_query(directory_path, query))
                
                # Cleanup if memory usage is high
                if current_memory > initial_memory * 2:
                    console.print("[yellow]Cleaning up cache...[/yellow]")
                
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
