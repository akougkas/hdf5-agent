"""
Single-File HDF5 Agent (SFA-HDF5)

This module implements a natural language interface for exploring HDF5 data files using
a local Language Model (LLM) served by Ollama. The agent provides functionality to:
- List and navigate HDF5 files, groups, and datasets
- Retrieve group and file attributes
- Retrieve dataset information, shapes, data types, and attributes
- Read and analyze data slices
- Generate dataset summaries
- Compare datasets across files

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
# version = "0.2.0"
# authors = ["Anthony Kougkas | https://akougkas.io"]
# description = "Single-File HDF5 Agent with natural language interface"
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
import json
import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional, List, Union
import argparse
import sys
from functools import lru_cache
import time
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

DEFAULT_MODEL = "phi4:latest"  # Default Ollama model
CACHE_SIZE = 128  # Configurable cache size for LRU caching
MAX_SLICE_SIZE = 1_000_000  # Max elements to read (~4MB for float32), adjustable
METRICS = {
    "tool_calls": {},
    "llm_calls": 0,
    "llm_time": 0.0,
    "token_count": 0
}
HISTORY = []  # Store command history for interactive mode
console = Console()

# --- Pydantic Models for Tool Arguments ---
class ListGroupsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")

class ListDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    group_path: str = Field(..., description="Path to the group within the file")

class GetGroupAttributeArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    group_path: str = Field(..., description="Path to the group within the file")
    attribute_name: str = Field(..., description="Name of the attribute to retrieve")

class GetFileAttributeArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    attribute_name: str = Field(..., description="Name of the attribute to retrieve")

class GetDatasetInfoArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    dataset_path: str = Field(..., description="Path to the dataset within the file")

class ReadDatasetDataArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    dataset_path: str = Field(..., description="Path to the dataset within the file")
    slice_start: Optional[List[int]] = Field(None, description="Start indices for slicing (e.g., [0] or [0, 0])")
    slice_end: Optional[List[int]] = Field(None, description="End indices for slicing (e.g., [10] or [5, 5])")

class SummarizeDatasetArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")
    dataset_path: str = Field(..., description="Path to the dataset within the file")

class CompareDatasetsArgs(BaseModel):
    file_path1: str = Field(..., description="Relative path to the first HDF5 file")
    dataset_path1: str = Field(..., description="Path to the first dataset")
    file_path2: str = Field(..., description="Relative path to the second HDF5 file")
    dataset_path2: str = Field(..., description="Path to the second dataset")

class ListAllDatasetsArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")

class GetFileMetadataArgs(BaseModel):
    file_path: str = Field(..., description="Relative path to the HDF5 file")

# --- Tool Functions with Timing ---
def list_files(directory_path: str) -> Dict[str, Any]:
    """List all HDF5 files in the specified directory."""
    start_time = time.time()
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        result = {"error": f"Invalid directory: {directory_path}"}
    else:
        files = [f.name for f in dir_path.glob("*.h5") if h5py.is_hdf5(f)]
        result = {"files": files, "directory": str(directory_path)} if files else {"error": "No HDF5 files found in the directory"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["list_files"] = METRICS["tool_calls"].get("list_files", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["list_files"]["count"] += 1
    METRICS["tool_calls"]["list_files"]["time"] += duration
    return result

def list_groups(directory_path: str, file_path: str) -> Dict[str, Any]:
    """List all groups in the specified HDF5 file."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                groups = []
                f.visit(lambda name: groups.append(name) if isinstance(f[name], h5py.Group) else None)
                result = {"groups": groups, "file": file_path}
        except Exception as e:
            result = {"error": f"Error listing groups: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["list_groups"] = METRICS["tool_calls"].get("list_groups", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["list_groups"]["count"] += 1
    METRICS["tool_calls"]["list_groups"]["time"] += duration
    return result

def list_datasets(directory_path: str, file_path: str, group_path: str) -> Dict[str, Any]:
    """List all datasets in the specified group within an HDF5 file."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if group_path not in f:
                    result = {"error": f"Group not found: {group_path}"}
                else:
                    datasets = []
                    f[group_path].visit(lambda name: datasets.append(f"{group_path}/{name}" if group_path != '/' else name) if isinstance(f[group_path][name], h5py.Dataset) else None)
                    result = {"datasets": datasets, "group": group_path}
        except Exception as e:
            result = {"error": f"Error listing datasets: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["list_datasets"] = METRICS["tool_calls"].get("list_datasets", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["list_datasets"]["count"] += 1
    METRICS["tool_calls"]["list_datasets"]["time"] += duration
    return result

@lru_cache(maxsize=CACHE_SIZE)
def get_group_attribute(directory_path: str, file_path: str, group_path: str, attribute_name: str) -> Dict[str, Any]:
    """Retrieve the value of a specific attribute of a group."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if group_path not in f:
                    result = {"error": f"Group not found: {group_path}"}
                elif not isinstance(f[group_path], h5py.Group):
                    result = {"error": f"Not a group: {group_path}"}
                elif attribute_name not in f[group_path].attrs:
                    result = {"error": f"Attribute '{attribute_name}' not found in group '{group_path}'"}
                else:
                    value = f[group_path].attrs[attribute_name]
                    result = {"attribute": attribute_name, "value": str(value), "group": group_path}
        except Exception as e:
            result = {"error": f"Error retrieving group attribute: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["get_group_attribute"] = METRICS["tool_calls"].get("get_group_attribute", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["get_group_attribute"]["count"] += 1
    METRICS["tool_calls"]["get_group_attribute"]["time"] += duration
    return result

@lru_cache(maxsize=CACHE_SIZE)
def get_file_attribute(directory_path: str, file_path: str, attribute_name: str) -> Dict[str, Any]:
    """Retrieve the value of a specific attribute of the HDF5 file."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if attribute_name not in f.attrs:
                    result = {"error": f"Attribute '{attribute_name}' not found in file '{file_path}'"}
                else:
                    value = f.attrs[attribute_name]
                    result = {"attribute": attribute_name, "value": str(value), "file": file_path}
        except Exception as e:
            result = {"error": f"Error retrieving file attribute: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["get_file_attribute"] = METRICS["tool_calls"].get("get_file_attribute", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["get_file_attribute"]["count"] += 1
    METRICS["tool_calls"]["get_file_attribute"]["time"] += duration
    return result

@lru_cache(maxsize=CACHE_SIZE)
def get_dataset_info(directory_path: str, file_path: str, dataset_path: str) -> Dict[str, Any]:
    """Retrieve metadata about a specific dataset (shape, dtype, attributes)."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if dataset_path not in f:
                    result = {"error": f"Dataset not found: {dataset_path}"}
                elif not isinstance(f[dataset_path], h5py.Dataset):
                    result = {"error": f"Not a dataset: {dataset_path}"}
                else:
                    ds = f[dataset_path]
                    attrs = dict(ds.attrs.items())
                    result = {
                        "dataset": dataset_path,
                        "shape": list(ds.shape),
                        "dtype": str(ds.dtype),
                        "attributes": {k: str(v) for k, v in attrs.items()}
                    }
        except Exception as e:
            result = {"error": f"Error retrieving dataset info: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["get_dataset_info"] = METRICS["tool_calls"].get("get_dataset_info", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["get_dataset_info"]["count"] += 1
    METRICS["tool_calls"]["get_dataset_info"]["time"] += duration
    return result

def read_dataset_data(directory_path: str, file_path: str, dataset_path: str, slice_start: Optional[List[int]] = None, slice_end: Optional[List[int]] = None) -> Dict[str, Any]:
    """Read data from a specific dataset with optional slicing, respecting MAX_SLICE_SIZE."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if dataset_path not in f:
                    result = {"error": f"Dataset not found: {dataset_path}"}
                elif not isinstance(f[dataset_path], h5py.Dataset):
                    result = {"error": f"Not a dataset: {dataset_path}"}
                else:
                    ds = f[dataset_path]
                    shape = ds.shape
                    
                    # Handle slicing
                    if slice_start is None and slice_end is None:
                        slice_size = np.prod(shape)
                        slice_obj = slice(None)
                    else:
                        slice_start = slice_start or [0] * len(shape)
                        slice_end = slice_end or list(shape)
                        if len(slice_start) != len(shape) or len(slice_end) != len(shape):
                            return {"error": f"Slice dimensions ({len(slice_start)}, {len(slice_end)}) must match dataset dimensions ({len(shape)})"}
                        slices = []
                        slice_size = 1
                        for i, (start, end, dim) in enumerate(zip(slice_start, slice_end, shape)):
                            start = max(0, min(start, dim))
                            end = max(start, min(end, dim))
                            slices.append(slice(start, end))
                            slice_size *= (end - start)
                        slice_obj = tuple(slices) if len(slices) > 1 else slices[0]
                    
                    if slice_size > MAX_SLICE_SIZE:
                        return {"error": f"Requested slice size ({slice_size} elements) exceeds MAX_SLICE_SIZE ({MAX_SLICE_SIZE})"}
                    
                    data = ds[slice_obj]
                    if isinstance(data, np.ndarray):
                        data_str = np.array2string(data, threshold=100, edgeitems=3, max_line_width=100)
                    else:
                        data_str = str(data)
                    
                    result = {
                        "dataset": dataset_path,
                        "data": data_str,
                        "shape": list(data.shape) if hasattr(data, 'shape') else [],
                        "dtype": str(data.dtype) if hasattr(data, 'dtype') else str(type(data).__name__)
                    }
        except Exception as e:
            result = {"error": f"Error reading dataset: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["read_dataset_data"] = METRICS["tool_calls"].get("read_dataset_data", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["read_dataset_data"]["count"] += 1
    METRICS["tool_calls"]["read_dataset_data"]["time"] += duration
    return result

def summarize_dataset(directory_path: str, file_path: str, dataset_path: str) -> Dict[str, Any]:
    """Provide a summary of a dataset's contents, including statistical analysis for numerical data or value summaries for strings."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                if dataset_path not in f:
                    result = {"error": f"Dataset not found: {dataset_path}"}
                elif not isinstance(f[dataset_path], h5py.Dataset):
                    result = {"error": f"Not a dataset: {dataset_path}"}
                else:
                    ds = f[dataset_path]
                    total_size = np.prod(ds.shape)
                    
                    if total_size == 0:
                        result = {"dataset": dataset_path, "summary": "Dataset is empty", "shape": list(ds.shape), "dtype": str(ds.dtype)}
                    elif total_size > MAX_SLICE_SIZE:
                        result = {"error": f"Dataset size ({total_size} elements) exceeds MAX_SLICE_SIZE ({MAX_SLICE_SIZE}) for summarization"}
                    else:
                        data = ds[()]
                        summary = {}
                        summary["shape"] = list(ds.shape)
                        summary["dtype"] = str(ds.dtype)
                        
                        if np.issubdtype(ds.dtype, np.number):
                            # Numerical data summary
                            summary["count"] = int(total_size)
                            summary["min"] = float(np.min(data))
                            summary["max"] = float(np.max(data))
                            summary["mean"] = float(np.mean(data))
                            summary["median"] = float(np.median(data))
                            summary["std"] = float(np.std(data))
                            summary["quantiles"] = {
                                "25%": float(np.percentile(data, 25)),
                                "50%": float(np.percentile(data, 50)),
                                "75%": float(np.percentile(data, 75))
                            }
                        elif ds.dtype.type is np.str_ or ds.dtype.type is np.bytes_:
                            # String data summary
                            data = data.astype(str) if ds.dtype.type is np.bytes_ else data
                            unique_values, counts = np.unique(data, return_counts=True)
                            summary["count"] = int(total_size)
                            summary["unique_values"] = int(len(unique_values))
                            top_indices = np.argsort(-counts)[:5]
                            summary["most_frequent"] = {str(unique_values[i]): int(counts[i]) for i in top_indices}
                            summary["sample"] = [str(data.flat[i]) for i in range(min(5, total_size))]
                        else:
                            # Generic summary for other types
                            summary["sample"] = str(data)[:1000]  # Truncate for brevity
                        
                        result = {"dataset": dataset_path, "summary": summary}
        except Exception as e:
            result = {"error": f"Error summarizing dataset: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["summarize_dataset"] = METRICS["tool_calls"].get("summarize_dataset", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["summarize_dataset"]["count"] += 1
    METRICS["tool_calls"]["summarize_dataset"]["time"] += duration
    return result

def compare_datasets(directory_path: str, file_path1: str, dataset_path1: str, file_path2: str, dataset_path2: str) -> Dict[str, Any]:
    """Compare two datasets from potentially different HDF5 files for equality or differences."""
    start_time = time.time()
    full_path1 = os.path.join(directory_path, file_path1)
    full_path2 = os.path.join(directory_path, file_path2)
    
    if not os.path.exists(full_path1):
        result = {"error": f"File not found: {file_path1}"}
    elif not os.path.exists(full_path2):
        result = {"error": f"File not found: {file_path2}"}
    else:
        try:
            with h5py.File(full_path1, 'r') as f1, h5py.File(full_path2, 'r') as f2:
                if dataset_path1 not in f1:
                    result = {"error": f"Dataset not found: {dataset_path1} in {file_path1}"}
                elif not isinstance(f1[dataset_path1], h5py.Dataset):
                    result = {"error": f"Not a dataset: {dataset_path1} in {file_path1}"}
                elif dataset_path2 not in f2:
                    result = {"error": f"Dataset not found: {dataset_path2} in {file_path2}"}
                elif not isinstance(f2[dataset_path2], h5py.Dataset):
                    result = {"error": f"Not a dataset: {dataset_path2} in {file_path2}"}
                else:
                    ds1 = f1[dataset_path1]
                    ds2 = f2[dataset_path2]
                    size1 = np.prod(ds1.shape)
                    size2 = np.prod(ds2.shape)
                    
                    if size1 > MAX_SLICE_SIZE or size2 > MAX_SLICE_SIZE:
                        result = {"error": f"Dataset size exceeds MAX_SLICE_SIZE ({MAX_SLICE_SIZE}): {size1} or {size2} elements"}
                    elif ds1.shape != ds2.shape:
                        result = {
                            "dataset1": {"file": file_path1, "path": dataset_path1, "shape": list(ds1.shape), "dtype": str(ds1.dtype)},
                            "dataset2": {"file": file_path2, "path": dataset_path2, "shape": list(ds2.shape), "dtype": str(ds2.dtype)},
                            "identical": False,
                            "difference": "Datasets have different shapes"
                        }
                    elif ds1.dtype != ds2.dtype:
                        result = {
                            "dataset1": {"file": file_path1, "path": dataset_path1, "shape": list(ds1.shape), "dtype": str(ds1.dtype)},
                            "dataset2": {"file": file_path2, "path": dataset_path2, "shape": list(ds2.shape), "dtype": str(ds2.dtype)},
                            "identical": False,
                            "difference": "Datasets have different data types"
                        }
                    else:
                        data1 = ds1[()]
                        data2 = ds2[()]
                        if np.issubdtype(ds1.dtype, np.number):
                            identical = np.allclose(data1, data2, equal_nan=True)
                            differences = "Values are approximately equal" if identical else "Numerical differences found"
                        else:
                            identical = np.array_equal(data1, data2)
                            differences = "Values are identical" if identical else "Values differ"
                        result = {
                            "dataset1": {"file": file_path1, "path": dataset_path1, "shape": list(ds1.shape), "dtype": str(ds1.dtype)},
                            "dataset2": {"file": file_path2, "path": dataset_path2, "shape": list(ds2.shape), "dtype": str(ds2.dtype)},
                            "identical": identical,
                            "difference": differences
                        }
        except Exception as e:
            result = {"error": f"Error comparing datasets: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["compare_datasets"] = METRICS["tool_calls"].get("compare_datasets", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["compare_datasets"]["count"] += 1
    METRICS["tool_calls"]["compare_datasets"]["time"] += duration
    return result

def list_all_datasets(directory_path: str, file_path: str) -> Dict[str, Any]:
    """List all datasets in the HDF5 file, grouped by their parent groups."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            with h5py.File(full_path, 'r') as f:
                datasets_by_group = {}
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        parent_group = '/' if '/' not in name else name.rsplit('/', 1)[0]
                        datasets_by_group.setdefault(parent_group, []).append(name)
                f.visititems(collect_datasets)
                result = {"datasets_by_group": datasets_by_group, "file": file_path}
        except Exception as e:
            result = {"error": f"Error listing all datasets: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["list_all_datasets"] = METRICS["tool_calls"].get("list_all_datasets", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["list_all_datasets"]["count"] += 1
    METRICS["tool_calls"]["list_all_datasets"]["time"] += duration
    return result

def get_file_metadata(directory_path: str, file_path: str) -> Dict[str, Any]:
    """Retrieve metadata stats like creation time and size for the HDF5 file."""
    start_time = time.time()
    full_path = os.path.join(directory_path, file_path)
    if not os.path.exists(full_path):
        result = {"error": f"File not found: {file_path}"}
    else:
        try:
            # File system metadata
            stat_info = os.stat(full_path)
            result = {
                "file": file_path,
                "size_bytes": stat_info.st_size,
                "creation_time": time.ctime(stat_info.st_ctime),
                "modification_time": time.ctime(stat_info.st_mtime)
            }
            # Add HDF5-specific metadata if available
            with h5py.File(full_path, 'r') as f:
                attrs = dict(f.attrs.items())
                if attrs:
                    result["attributes"] = {k: str(v) for k, v in attrs.items()}
        except Exception as e:
            result = {"error": f"Error retrieving file metadata: {str(e)}"}
    duration = time.time() - start_time
    METRICS["tool_calls"]["get_file_metadata"] = METRICS["tool_calls"].get("get_file_metadata", {"count": 0, "time": 0.0})
    METRICS["tool_calls"]["get_file_metadata"]["count"] += 1
    METRICS["tool_calls"]["get_file_metadata"]["time"] += duration
    return result

# --- Tool Registry ---
tool_registry = {
    "list_files": {"func": list_files, "args_model": None},
    "list_groups": {"func": list_groups, "args_model": ListGroupsArgs},
    "list_datasets": {"func": list_datasets, "args_model": ListDatasetsArgs},
    "get_group_attribute": {"func": get_group_attribute, "args_model": GetGroupAttributeArgs},
    "get_file_attribute": {"func": get_file_attribute, "args_model": GetFileAttributeArgs},
    "get_dataset_info": {"func": get_dataset_info, "args_model": GetDatasetInfoArgs},
    "read_dataset_data": {"func": read_dataset_data, "args_model": ReadDatasetDataArgs},
    "summarize_dataset": {"func": summarize_dataset, "args_model": SummarizeDatasetArgs},
    "compare_datasets": {"func": compare_datasets, "args_model": CompareDatasetsArgs},
    "list_all_datasets": {"func": list_all_datasets, "args_model": ListAllDatasetsArgs},
    "get_file_metadata": {"func": get_file_metadata, "args_model": GetFileMetadataArgs}
}

# --- Processing Agent ---
async def processing_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> Dict[str, Any]:
    """Process the query using tools and return structured results, handling multi-step queries."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        task = progress.add_task("Processing query...", total=None)
        accumulated_results = {}
        messages = [
            {
                "role": "system",
                "content": (
                    f"The HDF5 files are located in {directory_path}. Use relative paths (e.g., 'test_data.h5') without a leading '/'.\n\n"
                    f"Query: {query}\n\n"
                    "Respond only in JSON:\n"
                    "- {\"tool_call\": {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}} to execute a tool\n"
                    "- {\"done\": true, \"results\": <accumulated_results>} to finish with all results\n"
                    "- {\"error\": \"message\"} if the query cannot be processed\n\n"
                    "Tools:\n"
                    "- list_files: List HDF5 files (no arguments, use empty {})\n"
                    "- list_groups: List groups (argument: 'file_path')\n"
                    "- list_datasets: List datasets in a group (arguments: 'file_path', 'group_path'; use '/' for root)\n"
                    "- get_group_attribute: Get a group attribute (arguments: 'file_path', 'group_path', 'attribute_name')\n"
                    "- get_file_attribute: Get a file attribute (arguments: 'file_path', 'attribute_name')\n"
                    "- get_dataset_info: Get dataset metadata (arguments: 'file_path', 'dataset_path')\n"
                    "- read_dataset_data: Read dataset data (arguments: 'file_path', 'dataset_path', optional 'slice_start' and 'slice_end' as lists; limited to {MAX_SLICE_SIZE} elements)\n"
                    "- summarize_dataset: Summarize dataset contents (arguments: 'file_path', 'dataset_path'; provides statistics for numbers or value counts for strings; limited to {MAX_SLICE_SIZE} elements)\n"
                    "- compare_datasets: Compare two datasets for equality or differences (arguments: 'file_path1', 'dataset_path1', 'file_path2', 'dataset_path2'; limited to {MAX_SLICE_SIZE} elements per dataset)\n"
                    "- list_all_datasets: List all datasets grouped by parent groups (argument: 'file_path')\n"
                    "- get_file_metadata: Get file metadata like size and creation time (argument: 'file_path')\n\n"
                    "Rules:\n"
                    "- For multi-step queries, execute tools sequentially, accumulating results in each step.\n"
                    "- Use previous tool results (from 'accumulated_results') to inform subsequent tool calls.\n"
                    "- For 'read_dataset_data', 'summarize_dataset', and 'compare_datasets', respect the {MAX_SLICE_SIZE} element limit.\n"
                    "- When all parts of the query are resolved, return {\"done\": true, \"results\": <accumulated_results>}.\n"
                    "- Maintain state across iterations using 'accumulated_results'.\n\n"
                    "Examples:\n"
                    "- Query: 'First list groups in test.h5, then datasets in group1'\n"
                    "  Step 1: {\"tool_call\": {\"name\": \"list_groups\", \"arguments\": {\"file_path\": \"test.h5\"}}}\n"
                    "  Step 2: {\"tool_call\": {\"name\": \"list_datasets\", \"arguments\": {\"file_path\": \"test.h5\", \"group_path\": \"group1\"}}}\n"
                    "  Step 3: {\"done\": true, \"results\": {\"groups\": {...}, \"datasets\": {...}}}\n"
                    "- Query: 'Compare dataset data in test.h5 with data_backup in backup.h5'\n"
                    "  Step 1: {\"tool_call\": {\"name\": \"compare_datasets\", \"arguments\": {\"file_path1\": \"test.h5\", \"dataset_path1\": \"data\", \"file_path2\": \"backup.h5\", \"dataset_path2\": \"data_backup\"}}}\n"
                    "  Step 2: {\"done\": true, \"results\": {\"compare_datasets\": {...}}}\n"
                )
            }
        ]
        
        max_iterations = 10  # Increased to handle more steps
        for iteration in range(max_iterations):
            start_time = time.time()
            response = await client.chat(model=model, messages=messages, format="json")
            duration = time.time() - start_time
            METRICS["llm_calls"] += 1
            METRICS["llm_time"] += duration
            METRICS["token_count"] += response.get('eval_count', 0)
            
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
                messages.append({"role": "tool", "content": json.dumps({"tool": tool_name, "result": result})})
                if "error" not in result:
                    accumulated_results[tool_name] = result
                    messages.append({"role": "system", "content": f"Accumulated results so far: {json.dumps(accumulated_results)}\nContinue with the next step or finish if done."})
                else:
                    messages.append({"role": "system", "content": f"Tool failed: {result['error']}. Adjust or proceed."})
            elif "done" in data and "results" in data:
                progress.update(task, description="Query processed")
                return data["results"]
            elif "error" in data:
                progress.update(task, description="Processing failed")
                return {"error": data["error"]}
            else:
                messages.append({"role": "system", "content": "Provide 'tool_call', 'done', or 'error'."})
        
        progress.update(task, description="Processing failed")
        return {"error": "Processing failed after max iterations"}

# --- Interface Agent ---
async def interface_agent(directory_path: str, query: str, client: ollama.AsyncClient, model: str, processing_result: Dict[str, Any]) -> None:
    """Format and present the processing result conversationally."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        task = progress.add_task("Formatting response...", total=None)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly and helpful data exploration companion, acting as a guide to the HDF5 files in the provided directory. "
                    "Your role is to represent the contents of the files and assist the user with their queries in a natural, conversational way. "
                    "Focus strictly on the user's query and the processing result provided. "
                    "Do NOT suggest Python code snippets or external tools unless explicitly asked by the user. "
                    "Present results clearly and engagingly, using markdown for lists and structured data when appropriate. "
                    "If the result contains an error, explain it simply and suggest how I can assist further. "
                    "If the query is unclear or incomplete, ask the user for clarification or hint at my capabilities.\n\n"
                    f"Directory: {directory_path}\n"
                    f"Query: {query}\n"
                    f"Processing result: {json.dumps(processing_result)}\n\n"
                    "Respond with plain text, using markdown where appropriate."
                )
            }
        ]
        
        start_time = time.time()
        response = await client.chat(model=model, messages=messages)
        duration = time.time() - start_time
        METRICS["llm_calls"] += 1
        METRICS["llm_time"] += duration
        METRICS["token_count"] += response.get('eval_count', 0)
        
        print("\nHDF5-Agent:")
        print(response['message']['content'].strip())
        
        print("\n--- Performance Metrics ---")
        for tool, stats in METRICS["tool_calls"].items():
            avg_time = stats["time"] / stats["count"] if stats["count"] > 0 else 0
            print(f"{tool}: {stats['count']} calls, {stats['time']:.2f}s total, {avg_time:.2f}s avg")
        avg_llm_time = METRICS["llm_time"] / METRICS["llm_calls"] if METRICS["llm_calls"] > 0 else 0
        print(f"LLM: {METRICS['llm_calls']} calls, {METRICS['llm_time']:.2f}s total, {avg_llm_time:.2f}s avg")
        print(f"Tokens: {METRICS['token_count']}")
        progress.update(task, description="Response formatted")

# --- Query Runner ---
async def run_query(directory_path: str, query: str, client: ollama.AsyncClient, model: str) -> None:
    """Coordinate between Processing and Interface Agents."""
    processing_result = await processing_agent(directory_path, query, client, model)
    await interface_agent(directory_path, query, client, model, processing_result)

# --- Ollama Client Initialization ---
async def initialize_ollama_client(model: str) -> Optional[ollama.AsyncClient]:
    """Initialize the Ollama client and ensure the model is available."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
        task = progress.add_task("Initializing Ollama client...", total=None)
        try:
            client = ollama.AsyncClient(host='http://localhost:11434')
            models_response = await client.list()
            if 'models' not in models_response:
                raise ValueError("Unexpected response format: 'models' key not found")
            model_names = [m['model'] for m in models_response['models']]
            
            if model not in model_names:
                progress.update(task, description=f"Pulling model {model}...")
                await client.pull(model)
                updated_models = await client.list()
                model_names = [m['model'] for m in updated_models['models']]
                if model not in model_names:
                    raise RuntimeError(f"Failed to pull model {model}")
            progress.update(task, description="Ollama client initialized")
            return client
        except Exception as e:
            progress.update(task, description=f"Error: {str(e)}")
            print(f"Error initializing Ollama client: {type(e).__name__} - {str(e)}")
            return None

# --- Interactive Mode ---
async def interactive_mode(directory_path: str, client: ollama.AsyncClient, model: str):
    """Run an interactive shell with command history."""
    print("Entering interactive mode. Type 'exit' to quit, 'history' to see past commands, or a number to rerun a command.")
    while True:
        query = input("HDF5> ").strip()
        if query.lower() == "exit":
            HISTORY.clear()
            break
        elif query.lower() == "history":
            if not HISTORY:
                print("No command history yet.")
            else:
                for i, cmd in enumerate(HISTORY):
                    print(f"{i}: {cmd}")
        elif query.isdigit() and 0 <= int(query) < len(HISTORY):
            selected_query = HISTORY[int(query)]
            print(f"Re-running: {selected_query}")
            HISTORY.append(selected_query)
            await run_query(directory_path, selected_query, client, model)
        elif query:
            HISTORY.append(query)
            await run_query(directory_path, query, client, model)

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
        await interactive_mode(directory_path, client, model)

if __name__ == "__main__":
    asyncio.run(main())
