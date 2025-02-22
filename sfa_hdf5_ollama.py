"""
Single-File HDF5 Agent (SFA-HDF5)

This module implements a natural language interface for exploring HDF5 data files using
a local Language Model (LLM) served by Ollama. It uses an agentic flow with:
- Interface Agent: Handles user interaction and conversational output
- Processing Agent: Manages tool execution and structured responses

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
#     "pydantic>=2.4.2,<3.0.0",
#     "ollama~=0.4.7",
#     "numpy>=1.24.0,<2.0.0"
# ]
# requires-python = ">=3.8,<3.13"
# ///

import asyncio
import h5py
from pathlib import Path
import os
import json
import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional
import argparse
import sys

DEFAULT_MODEL = "granite3.1-dense:latest"  # Default Ollama model

# --- Pydantic Models for Tool Arguments ---
class ListGroupsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")

class ListDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    group_path: str = Field(..., description="Path to the group within the file")

# --- Tool Functions ---
def list_files(directory_path: str) -> Dict[str, Any]:
    """List all HDF5 files in the specified directory."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        return {"error": f"Invalid directory: {directory_path}"}
    files = [f.name for f in dir_path.glob("*.h5") if h5py.is_hdf5(f)]
    if not files:
        return {"error": "No HDF5 files found in the directory"}
    return {"files": files, "directory": str(directory_path)}

def list_groups(directory_path: str, file_path: str) -> Dict[str, Any]:
    """List all groups in the specified HDF5 file."""
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        return {"error": f"File not found: {file_path}"}
    try:
        with h5py.File(full_path, 'r') as f:
            groups = []
            f.visit(lambda name: groups.append(name) if isinstance(f[name], h5py.Group) else None)
            return {"groups": groups, "file": file_path}
    except Exception as e:
        return {"error": f"Error listing groups: {str(e)}"}

def list_datasets(directory_path: str, file_path: str, group_path: str) -> Dict[str, Any]:
    """List all datasets in the specified group within an HDF5 file."""
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        return {"error": f"File not found: {file_path}"}
    try:
        with h5py.File(full_path, 'r') as f:
            if group_path not in f:
                return {"error": f"Group not found: {group_path}"}
            datasets = []
            f[group_path].visit(lambda name: datasets.append(name) if isinstance(f[group_path][name], h5py.Dataset) else None)
            return {"datasets": datasets, "group": group_path}
    except Exception as e:
        return {"error": f"Error listing datasets: {str(e)}"}

# --- Tool Registry ---
tool_registry = {
    "list_files": {"func": list_files, "args_model": None},
    "list_groups": {"func": list_groups, "args_model": ListGroupsArgs},
    "list_datasets": {"func": list_datasets, "args_model": ListDatasetsArgs}
}

# --- Processing Agent ---
async def processing_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> Dict[str, Any]:
    """Process the query using tools and return structured results."""
    messages = [
        {
            "role": "system",
            "content": (
                f"The HDF5 files are located in {directory_path}. Use relative paths (e.g., 'test_data.h5') without a leading '/'.\n\n"
                f"Query: {query}\n\n"
                "Respond only in JSON:\n"
                "- {\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}}\n"
                "- {\"result\": <tool_result>} (return this immediately after a successful tool call)\n\n"
                "Tools:\n"
                "- list_files: List HDF5 files (no arguments, use empty {})\n"
                "- list_groups: List groups (argument: 'file_path')\n"
                "- list_datasets: List datasets (arguments: 'file_path', 'group_path'; use '/' for the root group)\n\n"
                "Rules:\n"
                "- Use 'list_files' only for listing all files, with empty arguments.\n"
                "- For specific files, use 'list_groups' or 'list_datasets' directly.\n"
                "- For queries about the 'root group', use 'list_datasets' with 'group_path': '/'.\n"
                "- If a tool call fails due to missing arguments, adjust and retry with correct arguments.\n"
                "- After a tool succeeds (no 'error' in result), return {\"result\": <tool_result>} and stop immediately."
            )
        }
    ]
    
    max_iterations = 5
    for _ in range(max_iterations):
        response = await client.chat(model=model, messages=messages, format="json")
        content = response['message']['content'].strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            messages.append({"role": "system", "content": "Invalid JSON. Provide valid JSON."})
            continue
        
        if "tool_call" in data:
            tool_call = data["tool_call"]
            tool_name = tool_call["name"]
            arguments = tool_call.get("arguments", {})
            
            print(f"Calling tool: {tool_name} with arguments: {arguments}")
            if tool_name not in tool_registry:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                tool = tool_registry[tool_name]
                if tool["args_model"]:
                    try:
                        args = tool["args_model"](**arguments)
                        result = tool["func"](directory_path, **args.model_dump())
                    except ValidationError as e:
                        result = {"error": f"Invalid arguments: {str(e)}"}
                else:
                    result = tool["func"](directory_path)
            
            print(f"Tool result: {result}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "tool", "content": json.dumps({"tool": tool_name, "result": result})})
            if "error" in result:
                if "group_path" in str(result.get("error", "")) and "required" in str(result.get("error", "")):
                    messages.append({"role": "system", "content": "Missing 'group_path'. For root group queries, use 'group_path': '/'."})
                else:
                    messages.append({"role": "system", "content": "Tool failed. Adjust or return an error result."})
            else:
                # Explicitly return the result and break after success
                return result
        elif "result" in data:
            return data["result"]
        else:
            messages.append({"role": "system", "content": "Provide 'tool_call' or 'result'."})
    
    return {"error": "Processing failed after max iterations"}

# --- Interface Agent ---
async def interface_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str, processing_result: Dict[str, Any]) -> None:
    """Format and present the processing result conversationally."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversational HDF5 file explorer. Present results naturally and engagingly.\n"
                "Avoid formal headers like 'Final response'. Offer next steps.\n\n"
                f"Directory: {directory_path}\n"
                f"Query: {query}\n"
                f"Processing result: {json.dumps(processing_result)}\n\n"
                " Respond with plain text, using markdown for lists."
            )
        }
    ]
    
    response = await client.chat(model=model, messages=messages)
    print("\nHDF5-Agent:")
    print(response['message']['content'].strip())

# --- Query Runner ---
async def run_query(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> None:
    """Coordinate between Processing and Interface Agents."""
    processing_result = await processing_agent(directory_path, query, client, model)
    await interface_agent(directory_path, query, client, model, processing_result)

# --- Ollama Client Initialization ---
async def initialize_ollama_client(model: str) -> Optional[ollama.AsyncClient]:
    """Initialize the Ollama client and ensure the model is available."""
    try:
        client = ollama.AsyncClient(host='http://localhost:11434')
        models_response = await client.list()
        if 'models' not in models_response:
            raise ValueError("Unexpected response format: 'models' key not found")
        model_names = [m['model'] for m in models_response['models']]
        
        if model not in model_names:
            print(f"Model {model} not found. Pulling from Ollama...")
            await client.pull(model)
            updated_models = await client.list()
            model_names = [m['model'] for m in updated_models['models']]
            if model not in model_names:
                raise RuntimeError(f"Failed to pull model {model}")
        return client
    except Exception as e:
        print(f"Error initializing Ollama client: {type(e).__name__} - {str(e)}")
        return None

# --- Main Function ---
async def main():
    """Parse arguments and run the HDF5 agent."""
    parser = argparse.ArgumentParser(description='HDF5 Agent with natural language interface')
    parser.add_argument('directory', help='Directory containing HDF5 files')
    parser.add_argument('query', nargs='?', help='Query to process (optional)')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Ollama model to use')
    args = parser.parse_args()
    
    directory_path = os.path.abspath(os.path.normpath(args.directory))
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        sys.exit(1)
    
    model = args.model
    client = await initialize_ollama_client(model)
    if client is None:
        print("Error: Could not initialize Ollama client. Exiting.")
        sys.exit(1)
    
    if args.query:
        await run_query(directory_path, args.query, client, model)
    else:
        print("Interactive mode not implemented yet.")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
