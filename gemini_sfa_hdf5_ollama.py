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
# version = "0.2.0"
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
from typing import Dict, Any, Optional, List
import argparse
import sys

DEFAULT_MODEL = "granite3.1-dense:latest"  # Default Ollama model

# --- Pydantic Models for Tool Arguments ---
class ListGroupsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")

class ListDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    group_path: str = Field(..., description="Path to the group within the file (e.g., '/' for root)")

# --- Tool Functions ---
def list_files(directory_path: str) -> Dict[str, Any]:
    """List all HDF5 files in the specified directory."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        return {"error": f"Invalid directory: {directory_path}"}
    files = [f.name for f in dir_path.glob("*.h5") if h5py.is_hdf5(f)]
    return {"files": files if files else [], "directory": str(directory_path)}

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
    """List datasets directly contained in the specified group within an HDF5 file."""
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        return {"error": f"File not found: {file_path}"}
    try:
        with h5py.File(full_path, 'r') as f:
            if group_path not in f:
                return {"error": f"Group not found: {group_path}"}
            group = f[group_path]
            datasets = [name for name in group if isinstance(group[name], h5py.Dataset)]
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
    """Process the query using tools and return structured results, supporting multi-step operations."""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert at exploring HDF5 files in {directory_path}. "
                f"Answer the user's query: {query}.\n\n"
                "Respond only in JSON:\n"
                "- {\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}}\n"
                "- {\"result\": <final_result>}\n\n"
                "Available tools:\n"
                "- list_files: List HDF5 files in the directory (no arguments)\n"
                "- list_groups: List all groups in a file (argument: 'file_path')\n"
                "- list_datasets: List datasets directly in a group (arguments: 'file_path', 'group_path'; use '/' for root)\n\n"
                "Guidelines:\n"
                "- Use relative paths for 'file_path' (e.g., 'test_data.h5').\n"
                "- For 'group_path', use '/' for the root group or the specific path (e.g., '/measurements').\n"
                "- For queries about specific groups, use 'list_datasets' with the exact 'group_path'.\n"
                "- For multi-step queries, use previous tool results (in messages) to inform next steps.\n"
                "- To find the deepest group, count '/' in group paths.\n"
                "- Return {\"result\": <final_result>} when all steps are complete.\n"
                "- Handle errors by reporting them in the result or adjusting the approach.\n\n"
                "Examples:\n"
                "Query: 'What datasets are available in the /measurements group of test_data.h5?'\n"
                "{\"tool_call\": {\"name\": \"list_datasets\", \"arguments\": {\"file_path\": \"test_data.h5\", \"group_path\": \"/measurements\"}}}\n"
                "Then: {\"result\": <tool_result>}\n\n"
                "Query: 'First show me all groups in test_data.h5, then list datasets in the deepest group'\n"
                "1. {\"tool_call\": {\"name\": \"list_groups\", \"arguments\": {\"file_path\": \"test_data.h5\"}}}\n"
                "2. After result, find deepest group, then: {\"tool_call\": {\"name\": \"list_datasets\", \"arguments\": {\"file_path\": \"test_data.h5\", \"group_path\": \"<deepest>\"}}}\n"
                "3. {\"result\": <datasets>}\n\n"
                "Query: 'Find all groups in test_data.h5 that contain datasets'\n"
                "1. List all groups, then call 'list_datasets' for each, collect groups with non-empty datasets.\n"
                "2. Return the list of groups.\n\n"
                "Debugging: Previous tool results are in the conversation history."
            )
        }
    ]
    
    max_iterations = 20  # Allow enough steps for complex queries
    for iteration in range(max_iterations):
        print(f"\n[Processing Agent] Iteration {iteration + 1}")
        response = await client.chat(model=model, messages=messages, format="json")
        content = response['message']['content'].strip()
        print(f"[LLM Response] {content}")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            messages.append({"role": "system", "content": "Invalid JSON. Respond with valid JSON."})
            continue
        
        if "tool_call" in data:
            tool_call = data["tool_call"]
            tool_name = tool_call["name"]
            arguments = tool_call.get("arguments", {})
            
            print(f"[Tool Call] {tool_name} with arguments: {arguments}")
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
            
            print(f"[Tool Result] {result}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "tool", "content": json.dumps(result)})
            messages.append({"role": "system", "content": "Tool call completed. Use the result for the next step or return the final result."})
        elif "result" in data:
            print(f"[Final Result] {data['result']}")
            return data["result"]
        else:
            messages.append({"role": "system", "content": "Provide a 'tool_call' or 'result' in JSON."})
    
    return {"error": "Query processing exceeded maximum iterations"}

# --- Interface Agent ---
async def interface_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str, processing_result: Dict[str, Any]) -> None:
    """Format and present the processing result conversationally."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly HDF5 file explorer assistant. Present the results naturally and conversationally.\n"
                "Use markdown for lists and clarity. Handle errors helpfully and suggest next steps.\n\n"
                f"Directory: {directory_path}\n"
                f"Query: {query}\n"
                f"Processing result: {json.dumps(processing_result)}\n\n"
                "Respond in plain text:"
            )
        }
    ]
    
    response = await client.chat(model=model, messages=messages)
    print("\nHDF5-Agent:")
    print(response['message']['content'].strip())

# --- Query Runner ---
async def run_query(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> None:
    """Coordinate between Processing and Interface Agents."""
    print(f"\n[Query] {query}")
    processing_result = await processing_agent(directory_path, query, client, model)
    await interface_agent(directory_path, query, client, model, processing_result)

# --- Ollama Client Initialization ---
async def initialize_ollama_client(model: str) -> Optional[ollama.AsyncClient]:
    """Initialize the Ollama client and ensure the model is available."""
    try:
        client = ollama.AsyncClient(host='http://localhost:11434')
        models_response = await client.list()
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
        print(f"Error initializing Ollama client: {str(e)}")
        return None

# --- Main Function ---
async def main():
    """Parse arguments and run the HDF5 agent with interactive mode."""
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
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("Enter your query: ")
            if query.lower() == 'exit':
                print("Exiting interactive mode.")
                break
            await run_query(directory_path, query, client, model)

if __name__ == "__main__":
    asyncio.run(main())
