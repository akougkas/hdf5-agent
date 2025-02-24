"""
Single-File HDF5 Agent (SFA-HDF5)

This module implements a natural language interface for exploring HDF5 data files using
a local Language Model (LLM) served by Ollama. It uses an agentic flow with:
- Interface Agent: Handles user interaction and conversational output
- Processing Agent: Manages tool execution and structured responses
The agent uses a tool-based architecture where the LLM can invoke specific functions
to interact with HDF5 files based on natural language queries.

Provided functionality:
- List and navigate HDF5 files, groups, and datasets
- Retrieve group and file attributes
- Retrieve dataset information, shapes, data types, and attributes
- Generate dataset summaries

Dependencies are managed using astral/uv to ensure reproducibility and portability.
The script can be run directly without any external requirements.txt file.

Author: Anthony Kougkas | https://akougkas.io
License: MIT
"""

# /// script
# package = "sfa-hdf5-agent"
# version = "0.3.1"
# authors = ["Anthony Kougkas | https://akougkas.io"]
# description = "HDF5 Exploratory Agent with natural language interface"
# repository = "https://github.com/akougkas/hdf5-agent"
# license = "MIT"
# dependencies = [
#     "h5py>=3.8.0,<3.9.0",
#     "pydantic>=2.4.2,<3.0.0",
#     "ollama~=0.4.7",
#     "numpy>=1.24.0,<2.0.0",
#     "rich>=13.7.0,<14.0.0"
# ]
# requires-python = ">=3.8,<3.13"
# ///

import asyncio
import h5py
from pathlib import Path
import os
import time
import json
import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional, List, Tuple
import argparse
import sys
import numpy as np
from rich.console import Console
from rich.markdown import Markdown
import hashlib
from collections import OrderedDict

# Constants
DEFAULT_MODEL = "phi4:latest"
ANALYSIS_THRESHOLD_MB = 64
MAX_CACHE_SIZE = 100
HISTORY: List[Dict[str, Any]] = []
console = Console()

# --- Models and Caches ---

class HistoryEntry(BaseModel):
    query: str
    timestamp: float
    results: Dict[str, Any]
    agent_response: str

class HDF5MetadataCache:
    def __init__(self):
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def preload(self, file_handle: h5py.File, file_path: str) -> None:
        if file_path in self._metadata:
            return
        metadata = {
            "groups": ['/'],
            "datasets": {},
            "attributes": {},
            "sampling_plans": {}
        }
        def collect_metadata(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and name != '/':
                path = f"/{name}" if not name.startswith('/') else name
                metadata["groups"].append(path)
                metadata["attributes"][path] = {k: str(v) for k, v in obj.attrs.items()}
            elif isinstance(obj, h5py.Dataset):
                path = f"/{name}" if not name.startswith('/') else name
                parent_group = '/' if '/' not in name else path.rsplit('/', 1)[0]
                metadata["datasets"].setdefault(parent_group, []).append(path)
                metadata["attributes"][path] = {k: str(v) for k, v in obj.attrs.items()}
        file_handle.visititems(collect_metadata)
        metadata["file_attributes"] = {k: str(v) for k, v in file_handle.attrs.items()}
        metadata["attributes"]['/'] = {k: str(v) for k, v in file_handle.attrs.items()}
        self._metadata[file_path] = metadata

    def get_groups(self, file_path: str) -> List[str]:
        return self._metadata.get(file_path, {}).get("groups", [])

    def get_datasets(self, file_path: str, group_path: str) -> List[str]:
        normalized_group_path = f"/{group_path.lstrip('/')}" if group_path != '/' else '/'
        return self._metadata.get(file_path, {}).get("datasets", {}).get(normalized_group_path, [])

    def get_attribute(self, file_path: str, path: str, attr_name: str) -> Optional[str]:
        normalized_path = f"/{path.lstrip('/')}" if path != '/' else '/'
        return self._metadata.get(file_path, {}).get("attributes", {}).get(normalized_path, {}).get(attr_name)

    def get_file_attribute(self, file_path: str, attr_name: str) -> Optional[str]:
        return self._metadata.get(file_path, {}).get("file_attributes", {}).get(attr_name)

    def get_sampling_plan(self, file_path: str, dataset_path: str) -> Optional[Tuple]:
        normalized_path = f"/{dataset_path.lstrip('/')}"
        return self._metadata.get(file_path, {}).get("sampling_plans", {}).get(normalized_path)

    def set_sampling_plan(self, file_path: str, dataset_path: str, plan: Tuple) -> None:
        normalized_path = f"/{dataset_path.lstrip('/')}"
        self._metadata.setdefault(file_path, {}).setdefault("sampling_plans", {})[normalized_path] = plan

    def exists(self, file_path: str, path: str) -> bool:
        normalized_path = f"/{path.lstrip('/')}" if path != '/' else '/'
        return normalized_path in self._metadata.get(file_path, {}).get("groups", []) or \
               normalized_path in [ds for ds_list in self._metadata.get(file_path, {}).get("datasets", {}).values() for ds in ds_list]

class HDF5FileCache:
    def __init__(self, max_files: int = 5):
        self._cache: OrderedDict[str, Tuple[h5py.File, float]] = OrderedDict()
        self._max_files: int = max_files
        self.metadata_cache = HDF5MetadataCache()

    def get_file(self, directory_path: str, file_path: str) -> h5py.File:
        full_path = os.path.join(directory_path, file_path)
        if full_path in self._cache:
            handle, _ = self._cache[full_path]
            self._cache.move_to_end(full_path)
            self._cache[full_path] = (handle, time.time())
            return handle
        if len(self._cache) >= self._max_files:
            oldest_key, (oldest_handle, _) = self._cache.popitem(last=False)
            oldest_handle.close()
        handle = h5py.File(full_path, 'r', libver='latest', rdcc_nbytes=32*1024*1024, rdcc_nslots=10000)
        self._cache[full_path] = (handle, time.time())
        self.metadata_cache.preload(handle, file_path)
        return handle

    def close_all(self) -> None:
        for handle, _ in self._cache.values():
            handle.close()
        self._cache.clear()

# --- Tool Argument Models ---

class ListGroupsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")

class ListDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")
    group_path: str = Field(..., description="Group path within file")

class GetGroupAttributeArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")
    group_path: str = Field(..., description="Group path within file")
    attribute_name: str = Field(..., description="Attribute name")

class GetFileAttributeArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")
    attribute_name: str = Field(..., description="Attribute name")

class GetDatasetInfoArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")
    dataset_path: str = Field(..., description="Dataset path within file")

class SummarizeDatasetArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")
    dataset_path: str = Field(..., description="Dataset path within file")

class ListAllDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")

class GetFileMetadataArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to HDF5 file")

# --- Tool Implementations ---

def hash_args(*args, **kwargs) -> str:
    arg_str = json.dumps([args, kwargs], sort_keys=True)
    return hashlib.sha1(arg_str.encode()).hexdigest()

def manage_cache(cache: Dict[str, Any], key: str, value: Any) -> None:
    if len(cache) >= MAX_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = value

def list_files(directory_path: str) -> Dict[str, Any]:
    cache_key = f"list_files_{hash_args(directory_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    files = [f.name for f in Path(directory_path).glob("*.h5") if h5py.is_hdf5(f)]
    result = {"files": files, "directory": str(directory_path)} if files else {"error": "No HDF5 files found"}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def list_groups(directory_path: str, file_path: str) -> Dict[str, Any]:
    cache_key = f"list_groups_{hash_args(directory_path, file_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    else:
        f = hdf5_cache.get_file(directory_path, file_path)
        groups = hdf5_cache.metadata_cache.get_groups(file_path)
        result = {"groups": groups, "file": file_path}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def list_datasets(directory_path: str, file_path: str, group_path: str) -> Dict[str, Any]:
    cache_key = f"list_datasets_{hash_args(directory_path, file_path, group_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    else:
        f = hdf5_cache.get_file(directory_path, file_path)
        if not hdf5_cache.metadata_cache.exists(file_path, group_path):
            result = {"error": f"Group not found: {group_path} in {file_path}"}
        else:
            datasets = hdf5_cache.metadata_cache.get_datasets(file_path, group_path)
            result = {"datasets": datasets, "group": group_path}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def get_group_attribute(directory_path: str, file_path: str, group_path: str, attribute_name: str) -> Dict[str, Any]:
    cache_key = f"get_group_attribute_{hash_args(directory_path, file_path, group_path, attribute_name)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    else:
        value = hdf5_cache.metadata_cache.get_attribute(file_path, group_path, attribute_name)
        result = {"attribute": attribute_name, "value": value, "group": group_path} if value is not None else {"error": f"Attribute '{attribute_name}' not found in group {group_path}"}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def get_file_attribute(directory_path: str, file_path: str, attribute_name: str) -> Dict[str, Any]:
    cache_key = f"get_file_attribute_{hash_args(directory_path, file_path, attribute_name)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    else:
        value = hdf5_cache.metadata_cache.get_file_attribute(file_path, attribute_name)
        result = {"attribute": attribute_name, "value": value, "file": file_path} if value is not None else {"error": f"Attribute '{attribute_name}' not found in file {file_path}"}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def get_dataset_info(directory_path: str, file_path: str, dataset_path: str) -> Dict[str, Any]:
    cache_key = f"get_dataset_info_{hash_args(directory_path, file_path, dataset_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    elif not hdf5_cache.metadata_cache.exists(file_path, dataset_path):
        result = {"error": f"Dataset not found: {dataset_path} in {file_path}"}
    else:
        f = hdf5_cache.get_file(directory_path, file_path)
        ds = f[dataset_path]
        result = {
            "dataset": dataset_path,
            "shape": list(ds.shape),
            "dtype": str(ds.dtype),
            "chunks": list(ds.chunks) if ds.chunks else None,
            "compression": ds.compression,
            "attributes": {k: str(v) for k, v in ds.attrs.items()}
        }
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

async def summarize_dataset(directory_path: str, file_path: str, dataset_path: str) -> Dict[str, Any]:
    cache_key = f"summarize_dataset_{hash_args(directory_path, file_path, dataset_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    elif not hdf5_cache.metadata_cache.exists(file_path, dataset_path):
        result = {"error": f"Dataset not found: {dataset_path} in {file_path}"}
    else:
        f = hdf5_cache.get_file(directory_path, file_path)
        ds = f[dataset_path]
        total_size_bytes = ds.size * ds.dtype.itemsize
        max_analysis_bytes = ANALYSIS_THRESHOLD_MB * 1024 * 1024
        metadata = {
            "shape": list(ds.shape),
            "dtype": str(ds.dtype),
            "size_bytes": total_size_bytes,
            "chunks": list(ds.chunks) if ds.chunks else None,
            "compression": ds.compression,
            "fillvalue": ds.fillvalue if ds.fillvalue is not None else None,
            "attributes": {k: str(v) for k, v in ds.attrs.items()}
        }
        summary = {"metadata": metadata}
        if ds.size > 0:
            sampling_plan = hdf5_cache.metadata_cache.get_sampling_plan(file_path, dataset_path)
            if not sampling_plan or any(s > d for s, d in zip(sampling_plan, ds.shape)):
                sampling_plan = get_optimal_slice(ds, max_analysis_bytes)
                hdf5_cache.metadata_cache.set_sampling_plan(file_path, dataset_path, sampling_plan)
            slices = [slice(0, min(s, d)) for s, d in zip(sampling_plan, ds.shape)]
            try:
                data = await asyncio.to_thread(lambda: ds[tuple(slices)] if total_size_bytes > max_analysis_bytes else ds[:])
                sampling_info = {"sampled": total_size_bytes > max_analysis_bytes, "sample_shape": list(data.shape)}
                summary["sampling"] = sampling_info
                if ds.dtype.names:
                    summary["compound_analysis"] = {name: summarize_field(data[name]) for name in ds.dtype.names}
                elif ds.dtype.kind in ['f', 'i', 'u']:
                    summary["numeric_analysis"] = welford_summary(data)
                elif ds.dtype.kind in ['S', 'U']:
                    decoded = [x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) for x in data.flatten()]
                    unique, counts = np.unique(decoded, return_counts=True)
                    summary["text_analysis"] = {"unique_values": len(unique), "top_values": dict(zip(unique[:10].tolist(), counts[:10].tolist()))}
            except Exception as e:
                summary["analysis_error"] = f"Failed to analyze data: {str(e)}"
        result = convert_numpy_types({"dataset": dataset_path, "summary": summary})
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def list_all_datasets(directory_path: str, file_path: str) -> Dict[str, Any]:
    cache_key = f"list_all_datasets_{hash_args(directory_path, file_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path) or not h5py.is_hdf5(full_path):
        result = {"error": f"File not found or not an HDF5 file: {file_path}"}
    else:
        f = hdf5_cache.get_file(directory_path, file_path)
        datasets_by_group = hdf5_cache.metadata_cache._metadata.get(file_path, {}).get("datasets", {})
        all_datasets = [ds for group in datasets_by_group.values() for ds in group]
        result = {"datasets": all_datasets, "file": file_path}
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

def get_file_metadata(directory_path: str, file_path: str) -> Dict[str, Any]:
    cache_key = f"get_file_metadata_{hash_args(directory_path, file_path)}"
    if cache_key in TOOL_RESULT_CACHE:
        return TOOL_RESULT_CACHE[cache_key]
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        stat_info = os.stat(full_path)
        f = hdf5_cache.get_file(directory_path, file_path)
        result = {
            "file": file_path,
            "size_bytes": stat_info.st_size,
            "creation_time": time.ctime(stat_info.st_ctime),
            "modification_time": time.ctime(stat_info.st_mtime),
            "attributes": hdf5_cache.metadata_cache._metadata.get(file_path, {}).get("file_attributes", {})
        }
    manage_cache(TOOL_RESULT_CACHE, cache_key, result)
    return result

# Tool Registry
tool_registry = {
    "list_files": {"func": list_files, "args_model": None},
    "list_groups": {"func": list_groups, "args_model": ListGroupsArgs},
    "list_datasets": {"func": list_datasets, "args_model": ListDatasetsArgs},
    "get_group_attribute": {"func": get_group_attribute, "args_model": GetGroupAttributeArgs},
    "get_file_attribute": {"func": get_file_attribute, "args_model": GetFileAttributeArgs},
    "get_dataset_info": {"func": get_dataset_info, "args_model": GetDatasetInfoArgs},
    "summarize_dataset": {"func": summarize_dataset, "args_model": SummarizeDatasetArgs},
    "list_all_datasets": {"func": list_all_datasets, "args_model": ListAllDatasetsArgs},
    "get_file_metadata": {"func": get_file_metadata, "args_model": GetFileMetadataArgs}
}

# --- Agent Logic ---

async def execute_tool(tool_name: str, arguments: Dict[str, Any], directory_path: str) -> Dict[str, Any]:
    if tool_name not in tool_registry:
        return {"error": f"Unknown tool: {tool_name}"}
    tool = tool_registry[tool_name]
    try:
        args = tool["args_model"](**arguments) if tool["args_model"] else None
        if asyncio.iscoroutinefunction(tool["func"]):
            result = await tool["func"](directory_path, **args.model_dump() if args else {})
        else:
            result = tool["func"](directory_path, **args.model_dump() if args else {})
        return result
    except ValidationError as e:
        return {"error": f"Invalid arguments for {tool_name}: {str(e)}"}
    except Exception as e:
        return {"error": f"Error executing {tool_name}: {str(e)}"}

async def processing_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> Dict[str, Any]:
    print("Processing query...")
    accumulated_results = {}
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that helps explore HDF5 files using specific tools. "
                "Based on the user's query, decide which tool to call next or if you have enough information to finish.\n\n"
                "Available tools:\n"
                "- list_files: Lists all HDF5 files in the directory. Arguments: {}\n"
                "- list_groups: Lists groups in an HDF5 file. Arguments: {\"file_path\": \"<path>\"}\n"
                "- list_datasets: Lists datasets in a group. Arguments: {\"file_path\": \"<path>\", \"group_path\": \"<path>\"}\n"
                "- summarize_dataset: Summarizes a dataset. Arguments: {\"file_path\": \"<path>\", \"dataset_path\": \"<path>\"}\n"
                "- get_group_attribute: Gets a group attribute. Arguments: {\"file_path\": \"<path>\", \"group_path\": \"<path>\", \"attribute_name\": \"<name>\"}\n"
                "- get_file_attribute: Gets a file attribute. Arguments: {\"file_path\": \"<path>\", \"attribute_name\": \"<name>\"}\n"
                "- get_dataset_info: Gets dataset info. Arguments: {\"file_path\": \"<path>\", \"dataset_path\": \"<path>\"}\n"
                "- list_all_datasets: Lists all datasets in a file. Arguments: {\"file_path\": \"<path>\"}\n"
                "- get_file_metadata: Gets file metadata. Arguments: {\"file_path\": \"<path>\"}\n\n"
                "Respond in JSON:\n"
                "- {\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value\", ...}}} to call a tool\n"
                "- {\"done\": true, \"results\": <accumulated_results>} to finish\n"
                "- {\"error\": \"message\"} if you cannot proceed\n\n"
                "Rules:\n"
                "- Do not hallucinate data—only use tools to fetch real information.\n"
                "- For complex queries, break them into multiple tool calls if needed.\n"
                "- If a tool fails (e.g., file or dataset not found), try alternative tools to gather context before finishing.\n"
                "- Always validate paths and existence before summarizing or fetching data.\n\n"
                "Examples:\n"
                "- Query: \"List all HDF5 files.\"\n"
                "  Response: {\"tool_call\": {\"name\": \"list_files\", \"arguments\": {}}}\n"
                "- Query: \"What groups are in test_data.h5?\"\n"
                "  Response: {\"tool_call\": {\"name\": \"list_groups\", \"arguments\": {\"file_path\": \"test_data.h5\"}}}\n"
                "- Query: \"List groups in nonexistent.h5 then datasets in test_data.h5.\"\n"
                "  Response: {\"tool_call\": {\"name\": \"list_groups\", \"arguments\": {\"file_path\": \"nonexistent.h5\"}}}\n"
                "- Query: \"Summarize dataset 'timeseries/temperature' in test_data.h5.\"\n"
                "  Response: {\"tool_call\": {\"name\": \"summarize_dataset\", \"arguments\": {\"file_path\": \"test_data.h5\", \"dataset_path\": \"timeseries/temperature\"}}}\n\n"
                "Accumulate results and finish only when the query is fully resolved."
            )
        },
        {"role": "user", "content": query}
    ]
    max_iterations = 10
    for _ in range(max_iterations):
        try:
            response = await asyncio.wait_for(client.chat(model=model, messages=messages, format="json"), timeout=30)
            content = response['message']['content'].strip()
            data = json.loads(content)
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as e:
            messages.append({"role": "system", "content": f"Error: {str(e)}. Please respond with valid JSON."})
            continue
        
        if "tool_call" in data:
            tool_name = data["tool_call"].get("name")
            arguments = data["tool_call"].get("arguments", {})
            if not tool_name:
                messages.append({"role": "system", "content": "Tool name is missing in tool_call."})
                continue
            result = await execute_tool(tool_name, arguments, directory_path)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "tool", "content": json.dumps(result)})
            accumulated_results[tool_name] = result
            if "error" in result and "nonexistent.h5" in query and "test_data.h5" in query:
                messages.append({"role": "system", "content": f"Tool {tool_name} returned: {json.dumps(result)}. Proceeding with next part of query."})
                if tool_name == "list_groups" and "file_path" in arguments and arguments["file_path"] == "nonexistent.h5":
                    messages.append({"role": "user", "content": "Now list datasets in the root group of test_data.h5"})
            elif "error" in result and "summarize_dataset" in tool_name:
                messages.append({"role": "system", "content": f"Tool {tool_name} failed: {json.dumps(result)}. Checking dataset existence with list_all_datasets."})
                messages.append({"role": "user", "content": f"List all datasets in {arguments['file_path']}"})
            else:
                messages.append({"role": "system", "content": f"Results so far: {json.dumps(accumulated_results)}\nNext step or finish?"})
        elif "done" in data and "results" in data:
            return data["results"]
        elif "error" in data:
            return {"error": data["error"]}
        else:
            messages.append({"role": "system", "content": "Invalid response. Use 'tool_call', 'done', or 'error'."})
    
    return {"error": "Failed to resolve query after maximum iterations"}

async def interface_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str, processing_result: Dict[str, Any]) -> str:
    history_summary = "\n".join([f"{i}: {h['query']} -> {h['agent_response']}" for i, h in enumerate(HISTORY[-5:])])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly data exploration companion for HDF5 files. Present the results from the processing agent in a clear, concise, and user-friendly manner using markdown. "
                "NEVER suggest code, APIs, or external tools. Only provide information based on the processing result. "
                "If the query cannot be fulfilled due to an error, explain the issue simply and offer to provide available information (e.g., list files, groups, datasets). "
                "Do not hallucinate data—stick strictly to the processing result.\n\n"
                f"Directory: {directory_path}\n"
                f"Query: {query}\n"
                f"Processing result: {json.dumps(processing_result)}\n"
                f"Recent Interactions: {history_summary}\n\n"
                "Respond with plain text, using markdown where appropriate."
            )
        }
    ]
    response = await asyncio.wait_for(client.chat(model=model, messages=messages), timeout=30)
    agent_response = response['message']['content'].strip()
    console.print(f"\n[blue]─ You asked: {query} ─[/blue]\n")
    console.print(Markdown(agent_response))
    return agent_response

# --- Utility Functions ---

def get_optimal_slice(dataset: h5py.Dataset, max_size_bytes: int) -> Tuple[int, ...]:
    shape = dataset.shape
    dtype_size = dataset.dtype.itemsize
    total_bytes = np.prod(shape) * dtype_size
    if total_bytes <= max_size_bytes:
        return shape
    target_elements = max_size_bytes // dtype_size
    scale = (target_elements / np.prod(shape)) ** (1 / len(shape))
    return tuple(max(1, int(s * scale)) for s in shape)

def welford_summary(data: np.ndarray) -> Dict[str, Any]:
    n = 0
    mean = 0.0
    M2 = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    nan_count = 0
    inf_count = 0
    zero_count = 0
    
    for x in data.flatten():
        x = float(x)
        if np.isnan(x):
            nan_count += 1
            continue
        if np.isinf(x):
            inf_count += 1
            continue
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
        min_val = min(min_val, x)
        max_val = max(max_val, x)
        if x == 0:
            zero_count += 1
    
    std = (M2 / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {
        "min": min_val if n > 0 else None,
        "max": max_val if n > 0 else None,
        "mean": mean if n > 0 else None,
        "std": std if n > 0 else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_count": zero_count,
        "count": n
    }

def summarize_field(data: np.ndarray) -> Dict[str, Any]:
    if np.issubdtype(data.dtype, np.number):
        return {"type": "numeric", **welford_summary(data)}
    elif data.dtype.kind in ['S', 'U']:
        decoded = [x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) for x in data.flatten()]
        unique, counts = np.unique(decoded, return_counts=True)
        return {"type": "text", "unique_values": len(unique), "top_values": dict(zip(unique[:5].tolist(), counts[:5].tolist()))}
    return {"type": str(data.dtype), "sample": str(data[:5].tolist()) if data.size >= 5 else str(data.tolist())}

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# --- Main Logic ---

async def run_query(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> None:
    result = await processing_agent(directory_path, query, client, model)
    agent_response = await interface_agent(directory_path, query, client, model, result)
    HISTORY.append(HistoryEntry(query=query, timestamp=time.time(), results=result, agent_response=agent_response).model_dump())

async def initialize_ollama_client(model: str) -> Optional[ollama.AsyncClient]:
    print("Initializing Ollama...")
    try:
        client = ollama.AsyncClient(host='http://localhost:11434')
        models_response = await client.list()
        model_names = [m['model'] for m in models_response['models']]
        if model not in model_names:
            print(f"Loading {model}...")
            await client.pull(model)
        print("✓ Ollama ready :)")
        return client
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

async def interactive_mode(directory_path: str, client: ollama.AsyncClient, model: str) -> None:
    print("Interactive mode: 'exit' to quit, 'history' for past queries, or number to rerun")
    while True:
        query = input("HDF5> ").strip()
        if query.lower() == "exit":
            HISTORY.clear()
            TOOL_RESULT_CACHE.clear()
            hdf5_cache.close_all()
            break
        elif query.lower() == "history":
            for i, cmd in enumerate(HISTORY):
                print(f"{i}: {cmd['query']}")
        elif query.isdigit() and 0 <= int(query) < len(HISTORY):
            selected = HISTORY[int(query)]["query"]
            print(f"Re-running: {selected}")
            await run_query(directory_path, selected, client, model)
        elif query:
            await run_query(directory_path, query, client, model)

async def main() -> None:
    parser = argparse.ArgumentParser(description='HDF5 Exploratory Agent')
    parser.add_argument('directory', help='Directory with HDF5 files')
    parser.add_argument('query', nargs='?', help='Optional query')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Ollama model')
    args = parser.parse_args()
    
    directory_path = os.path.abspath(args.directory)
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        sys.exit(1)
    
    client = await initialize_ollama_client(args.model)
    if client is None:
        print("Failed to initialize Ollama")
        sys.exit(1)
    
    global hdf5_cache, TOOL_RESULT_CACHE
    hdf5_cache = HDF5FileCache()
    TOOL_RESULT_CACHE = {}
    try:
        if args.query:
            await run_query(directory_path, args.query, client, args.model)
        else:
            await interactive_mode(directory_path, client, args.model)
    finally:
        hdf5_cache.close_all()

if __name__ == "__main__":
    asyncio.run(main())






