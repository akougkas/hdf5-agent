# SFA-HDF5: Single-File HDF5 Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, single-file Python agent for exploring and analyzing HDF5 files using natural language queries. This agent leverages a local Language Model (LLM) via Ollama to provide an intuitive, conversational interface for interacting with HDF5 data.

## Concept: The "Smart Folder"

The core idea is the "smart folder": place `sfa_hdf5_ollama.py` in a directory with your HDF5 data to create a self-contained, portable data exploration unit. This folder can be shared and used on any system with Python, `uv`, and Ollama installed, enabling seamless interaction with HDF5 files.

## Features

- **Self-contained:** Entire agent resides in a single Python file (`sfa_hdf5_ollama.py`).
- **HDF5 File Exploration:** Navigate files, groups, datasets, and their metadata.
- **Metadata Retrieval:** Access group and file attributes, plus dataset details (shape, dtype, attributes).
- **Dataset Summarization:** Generate statistical summaries for numerical data or value counts for strings (up to 64MB threshold).
- **Performance Optimization:** Caches metadata and tool results with a configurable LRU cache (default size: 100).
- **Local LLM Support:** Uses Ollama for local LLM execution (default model: `phi4:latest`).
- **Dependency Management:** Employs `uv` for automatic dependency installation via the `# /// script` block.
- **Interactive and CLI Modes:** Supports command-line queries and interactive chat (type `exit` to quit, `history` for past queries).
- **Multi-Step Queries:** Handles complex, multi-step explorations (e.g., list groups, then datasets).
- **Robust Error Handling:** Gracefully manages invalid files, paths, and query errors with descriptive feedback.

## Available Agent Tools

The agent uses a tool-based architecture with two agents:
- **Processing Agent**: Executes tools and processes structured queries in JSON format.
- **Interface Agent**: Formats results conversationally using markdown for readability.

Available agent tools:
1. **list_files**: Lists all HDF5 files in the directory.
2. **list_groups**: Lists all groups within an HDF5 file.
3. **list_datasets**: Lists datasets within a specified group (e.g., root '/' or '/measurements').
4. **get_group_attribute**: Retrieves a specific attribute from a group (e.g., 'description' from '/metadata').
5. **get_file_attribute**: Retrieves a specific attribute from the file (e.g., 'creation_date').
6. **get_dataset_info**: Retrieves dataset metadata (shape, data type, chunks, compression, attributes).
7. **summarize_dataset**: Summarizes dataset contents (stats for numbers, counts for strings; up to 64MB).
8. **list_all_datasets**: Lists all datasets in a file, grouped by parent groups.
9. **get_file_metadata**: Retrieves file metadata (size, creation/modification times, attributes).

*Note:* Tools like `read_dataset_data` and `compare_datasets` were removed in v0.3.0 to focus on exploration and summarization, aligning with performance and correctness goals.

## Requirements

- Python 3.8+
- `uv` (Universal Python Package Installer): [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- Ollama: [https://ollama.ai/](https://ollama.ai/)

## Installation

1. **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. **Install and start Ollama:**
    - Follow the instructions at [Ollama](https://ollama.ai/)
    - Pull the default model:
      ```bash
      ollama pull phi4:latest
      ```
3. **Place `sfa_hdf5_ollama.py` in your data directory:** Copy the script into the folder containing your HDF5 files.

## Usage

Run the script using `uv run`, which automatically installs dependencies from the `# /// script` block.

### Command-line Mode

```bash
uv run sfa_hdf5_ollama.py <directory_path> "<your_query>"
```
- `<directory_path>`: Path to the directory with HDF5 files and the script.
- `<your_query>`: Natural language query (optional).

**Examples:**
```bash
uv run sfa_hdf5_ollama.py data "List all HDF5 files in this directory"
uv run sfa_hdf5_ollama.py data "What groups are in test_data.h5?"
uv run sfa_hdf5_ollama.py data "List datasets in the / group of test_data.h5"
uv run sfa_hdf5_ollama.py data "Summarize dataset 'timeseries/temperature' in test_data.h5"
uv run sfa_hdf5_ollama.py data "Get the 'creation_date' attribute of file 'test_data.h5'"
```

### With Custom Models
- `directory`: (Required) Directory containing HDF5 files
- `query`: (Optional) Query to process; if omitted, enters interactive mode
- `-m, --model`: (Optional) Specify an Ollama model (default: phi4:latest)

**Example with custom model:**
```bash
uv run sfa_hdf5_ollama.py -m mistral data "What datasets exist in test_data.h5?"
```

### Interactive Mode
Run without a query to enter interactive mode:
```bash
uv run sfa_hdf5_ollama.py data
```
Type your query at the `HDF5>` prompt. Use `exit` to quit, `history` to view past queries, or a number to rerun a previous query.

**Example Interactive Session:**

```
hdf5-agent
uv run sfa_hdf5_ollama.py data
Initializing Ollama...
✓ Ollama ready :)
Interactive mode: 'exit' to quit, 'history' for past queries, or number to rerun
HDF5> how many groups are there in teh test_data.h5 file?         
Processing query...

─ You asked: how many groups are there in teh test_data.h5 file? ─

In the test_data.h5 file located at /mnt/nfs/dev/single-file-agents/hdf5-agent/data, there are a total of 9 groups.                                                                                               

If you need any more details about these groups or anything else regarding this HDF5 file, feel free to ask!                                                                                                      
HDF5> show me all groups
Processing query...

─ You asked: show me all groups ─

It seems like there was an issue processing your request because it wasn't clear whether you were asking about a specific directory or file for listing all HDF5 files and their respective groups.               

To help clarify:                                                                                                                                                                                                  

 • If you have a particular HDF5 file in mind (like test_data.h5), I can assist with details about that file, such as the number of groups within it.                                                             
 • For example, from our recent interactions, I noted that in the test_data.h5 file located at /mnt/nfs/dev/single-file-agents/hdf5-agent/data, there are a total of 9 groups.                                    

If you'd like to proceed with details on specific files or need assistance listing available files within a directory, just let me know!                                                                          
HDF5> show me all groups in 'test_data.h5' file?
Processing query...

─ You asked: show me all groups in 'test_data.h5' file? ─

Certainly! Here's the information regarding the groups in the test_data.h5 file:                                                                                                                                  

                                                                                              Groups in test_data.h5                                                                                              

In the HDF5 file named test_data.h5, located at /mnt/nfs/dev/single-file-agents/hdf5-agent/data, you will find the following groups:                                                                              

 • Root Group: /                                                                                                                                                                                                  
 • Image Data Group: /images                                                                                                                                                                                      
 • Nested Groups:                                                                                                                                                                                                 
    • /nested                                                                                                                                                                                                     
    • /nested/level1                                                                                                                                                                                              
    • /nested/level1/level2                                                                                                                                                                                       
    • /nested/level1/level2/level3                                                                                                                                                                                
 • Numerical Data Group: /numerical_data                                                                                                                                                                          
 • Text Data Group: /text_data                                                                                                                                                                                    
 • Time Series Data Group: /timeseries                                                                                                                                                                            

In total, there are 9 groups within this file.                                                                                                                                                                    

If you need more details about any specific group or have other questions regarding the test_data.h5 file, feel free to ask!                                                                                      
HDF5> summarize the dataset '/timeseries/temperature' in the file 'test_data.h5'
Processing query...

─ You asked: summarize the dataset '/timeseries/temperature' in the file 'test_data.h5' ─

                                                                         Summary of Dataset /timeseries/temperature in File test_data.h5                                                                          

The dataset /timeseries/temperature contains temperature data recorded over a period. Here's a detailed summary based on the processing results:                                                                  

                                                                                                    Metadata:                                                                                                     

 • Shape: [31]                                                                                                                                                                                                    
    • This indicates that there are 31 elements (data points) in this dataset.                                                                                                                                    
 • Data Type: float64                                                                                                                                                                                             
    • Each element is represented as a double-precision floating-point number.                                                                                                                                    
 • Size: 248 bytes                                                                                                                                                                                                
    • The total size of the data in memory is 248 bytes.                                                                                                                                                          
 • Chunks: None                                                                                                                                                                                                   
    • The dataset is stored without chunking, meaning it's read or written in one piece rather than divided into smaller chunks.                                                                                  
 • Compression: None                                                                                                                                                                                              
    • No compression technique has been applied to this dataset.                                                                                                                                                  
 • Fill Value: 0.0                                                                                                                                                                                                
    • A fill value of 0.0 is specified for missing data points.                                                                                                                                                   
 • Attributes:                                                                                                                                                                                                    
    • Sampling Rate: "1 day"                                                                                                                                                                                      
       • Data is sampled daily.                                                                                                                                                                                   
    • Start Date: "2024-01-01"                                                                                                                                                                                    
       • The first recorded date corresponds to January 1, 2024.                                                                                                                                                  
    • Unit: "Celsius"                                                                                                                                                                                             
       • Temperature values are given in degrees Celsius.                                                                                                                                                         

                                                                                                    Sampling:                                                                                                     

 • Sampled: False                                                                                                                                                                                                 
    • No subsampling of data points has been performed for analysis.                                                                                                                                              
 • Sample Shape: [31]                                                                                                                                                                                             
    • The shape is consistent with the original dataset, as no sampling was applied.                                                                                                                              

                                                                                                Numeric Analysis:                                                                                                 

 • Minimum Value: 13.484297667432232                                                                                                                                                                              
    • The lowest temperature recorded in this dataset.                                                                                                                                                            
 • Maximum Value: 25.91422947866164                                                                                                                                                                               
    • The highest temperature recorded.                                                                                                                                                                           
 • Mean (Average): 20.03770084486556                                                                                                                                                                              
    • The average temperature over the sampled period.                                                                                                                                                            
 • Standard Deviation: 3.6640272797334172                                                                                                                                                                         
    • A measure of variation or dispersion from the mean.                                                                                                                                                         
 • NaN Count: 0                                                                                                                                                                                                   
    • There are no missing values (NaNs) in the dataset.                                                                                                                                                          
 • Infinity Count: 0                                                                                                                                                                                              
    • No infinite values are present.                                                                                                                                                                             
 • Zero Count: 0                                                                                                                                                                                                  
    • No zero values recorded in this dataset.                                                                                                                                                                    
 • Count: 31                                                                                                                                                                                                      
    • The total number of data points is 31, matching the shape.                                                                                                                                                  

If you have any further questions or need more details about the dataset or other aspects of the HDF5 file, feel free to ask!                                                                                     
HDF5> which group is the most interesting?
Processing query...

─ You asked: which group is the most interesting? ─

The query "which group is the most interesting?" in the context of an HDF5 file requires specific criteria to be defined because what makes a group "interesting" can vary greatly depending on personal or       
research interests.                                                                                                                                                                                               

Explanation of the Error:                                                                                                                                                                                         

 • Subjectivity: The term "most interesting" is subjective and lacks clear, measurable criteria. Without additional information about what characteristics make a group stand out (e.g., specific datasets,       
   metadata, attributes), it's not possible to determine which group might be considered most interesting.                                                                                                        
 • Request for Details: To proceed with such a query, you would need to specify the file path and provide particular features or criteria that define "interesting" in this context.                              

Assistance Offered:                                                                                                                                                                                               

I can assist by providing details about specific files, groups, datasets, or metadata. For example:                                                                                                               

 1 If you have a particular HDF5 file, like test_data.h5, I can list all the groups within it.                                                                                                                    
 2 You could specify certain attributes or characteristics (e.g., large datasets, unique metadata) to help identify interesting groups.                                                                           

If you provide more details about what makes a group "interesting," I'll be happy to help analyze the available data accordingly. For instance, I have information on the structure and content of test_data.h5 if
that helps in narrowing down your query.                                                                                                                                                                          
HDF5> ^ exit
```

## Test Cases

### Basic Exploration
```bash
# List files
uv run sfa_hdf5_ollama.py data "List all HDF5 files"

# List groups
uv run sfa_hdf5_ollama.py data "What groups are in test_data.h5?"

# List datasets in root group
uv run sfa_hdf5_ollama.py data "List all datasets in the root group of test_data.h5"
```

### Metadata Retrieval
```bash
# Group attribute
uv run sfa_hdf5_ollama.py data "What is the 'description' attribute of group '/metadata' in test_data.h5?"

# File attribute
uv run sfa_hdf5_ollama.py data "What is the 'creation_date' attribute of file test_data.h5?"

# Dataset info
uv run sfa_hdf5_ollama.py data "Get the shape and data type of dataset 'data' in test_data.h5"
```

### Dataset Summarization
```bash
# Summarize dataset
uv run sfa_hdf5_ollama.py data "Summarize dataset 'timeseries/temperature' in test_data.h5"
```

### Complex Queries
```bash
# Multi-step exploration
uv run sfa_hdf5_ollama.py data "Show me all groups in test_data.h5, then list datasets in the first group"

# List all datasets
uv run sfa_hdf5_ollama.py data "What datasets exist in test_data.h5?"
```

### Error Handling
```bash
# Invalid file
uv run sfa_hdf5_ollama.py data "List groups in nonexistent.h5"

# Invalid group path
uv run sfa_hdf5_ollama.py data "List datasets in /not/a/real/path within test_data.h5"
```

## Dependencies
Defined in the `# /// script` block and installed by `uv`:
```text
h5py>=3.8.0,<3.9.0
pydantic>=2.4.2,<3.0.0
ollama~=0.4.7
numpy>=1.24.0,<2.0.0
rich>=13.7.0,<14.0.0
```

## Debugging
The script uses print-based logging (e.g., "Processing query...") for transparency. Review console output to trace execution and identify issues.

## Contributing
Contributions are welcome! Please submit pull requests or issues to the GitHub repository: https://github.com/akougkas/hdf5-agent.

## License
This project is licensed under the MIT License.
