# SFA-HDF5: Single-File HDF5 Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, single-file Python agent for exploring and analyzing HDF5 files using natural language queries. This agent leverages a local Language Model (LLM) via Ollama to provide an intuitive, conversational interface for interacting with HDF5 data.

## Concept: The "Smart Folder"

The core idea is the "smart folder": place `sfa_hdf5_ollama.py` in a directory with your HDF5 data to create a self-contained, portable data exploration unit. This folder can be shared and used on any system with Python, `uv`, and Ollama installed, enabling seamless interaction with HDF5 files.

## Features

- **Self-contained:** The entire agent resides in a single Python file.
- **HDF5 File Exploration:** Navigate HDF5 file structures (groups, datasets) with ease.
- **Data Analysis:** Retrieve dataset information directly within specified groups.
- **Local LLM Support:** Uses Ollama for local LLM execution (default model: `granite3.1-dense:latest`).
- **Dependency Management:** Employs `uv` for automatic dependency installation.
- **Interactive and CLI Modes:** Supports both command-line queries and an interactive chat mode.
- **Multi-Step Queries:** Handles complex, multi-step explorations (e.g., listing groups then datasets in the deepest group).
- **Robust Error Handling:** Gracefully manages invalid files, groups, and complex error cases.

## Available Tools

The agent uses a tool-based architecture with two agents:
- **Processing Agent**: Executes tools and processes structured queries.
- **Interface Agent**: Formats results conversationally.

Available tools:
1. **list_files**: Lists all HDF5 files in the directory.
2. **list_groups**: Lists all groups within an HDF5 file.
3. **list_datasets**: Lists datasets directly within a specified group (e.g., root '/' or '/measurements').

## Requirements

- Python 3.8+
- `uv` (Universal Python Package Installer): [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- Ollama: [https://ollama.ai/](https://ollama.ai/)

## Installation

1. **Install `uv`:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install and start Ollama**: Follow the instructions at [Ollama](https://ollama.ai/).
3. **Pull the default model (granite3.1-dense:latest)**:
```bash
ollama pull granite3.1-dense:latest
```
4. **Place sfa_hdf5_ollama.py in your data directory**: Copy the script into the folder containing your HDF5 files.


## Usage

Run the script using `uv run`, which automatically installs dependencies from the `# ///` script block.

### Command-line Mode

```bash
uv run sfa_hdf5_ollama.py <directory_path> "<your_query>"
```
- `<directory_path>`: Path to the directory with HDF5 files and the script.
- `<your_query>`: Natural language query.

Examples:
```bash
uv run sfa_hdf5_ollama.py data "List all HDF5 files in this directory"
uv run sfa_hdf5_ollama.py data "What datasets are available in the /measurements group of test_data.h5?"
uv run sfa_hdf5_ollama.py data "First show me all groups in test_data.h5, then list the datasets in the deepest group"
```

### Interactive Mode

Run without a query to enter interactive mode:
```bash
uv run sfa_hdf5_ollama.py data
```
Prompt: Enter your query:
Type `exit` to quit.

### Command-line Arguments

- `directory`: (Required) Directory containing HDF5 files.
- `query`: (Optional) Query to process; if omitted, enters interactive mode.
- `-m`, `--model`: (Optional) Specify an Ollama model (default: `granite3.1-dense:latest`).

Example with custom model:
```bash
uv run sfa_hdf5_ollama.py -m mistral data "What groups are in test_data.h5?"
```

## Supported Test Cases

### Basic Exploration

List datasets in root group:
```bash
uv run sfa_hdf5_ollama.py data "List all datasets in the root group of test_data.h5"
```

### Specific Group Datasets

Datasets in a group:
```bash
uv run sfa_hdf5_ollama.py data "What datasets are available in the /measurements group of test_data.h5?"
```

### Complex Queries

Multi-step exploration:
```bash
uv run sfa_hdf5_ollama.py data "First show me all groups in test_data.h5, then list the datasets in the deepest group"
```

Conditional exploration:
```bash
uv run sfa_hdf5_ollama.py data "Find all groups in test_data.h5 that contain datasets"
```

### Error Handling

Invalid file:
```bash
uv run sfa_hdf5_ollama.py data "Show me the contents of nonexistent.h5"
```

Invalid group path:
```bash
uv run sfa_hdf5_ollama.py data "List datasets in /not/a/real/path within test_data.h5"
```

Complex error case:
```bash
uv run sfa_hdf5_ollama.py data "First list groups in nonexistent.h5, then show datasets in test_data.h5"
```

## Dependencies

Dependencies are defined in the `# ///` script block and installed by `uv`:

```text
h5py>=3.8.0,<3.9.0
pydantic>=2.4.2,<3.0.0
ollama~=0.4.7
numpy>=1.24.0,<2.0.0
```

## Debugging

The Processing Agent logs each iteration, LLM response, tool call, and result with print statements prefixed by `[Processing Agent]`, `[LLM Response]`, `[Tool Call]`, and `[Tool Result]`. Review these logs to trace execution and identify issues.

## Contributing

Contributions are welcome! Please submit pull requests or issues to the GitHub repository.

## License

This project is licensed under the MIT License.
