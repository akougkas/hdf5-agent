# SFA-HDF5: Single-File HDF5 Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, single-file Python agent for exploring and analyzing HDF5 files using natural language queries.  This agent leverages a local Language Model (LLM) via Ollama to provide an intuitive interface for interacting with HDF5 data.

## Concept: The "Smart Folder"

The core idea is the "smart folder": place `sfa_hdf5_ollama.py` in a directory with your HDF5 data to create a self-contained, portable data exploration unit. This folder can be shared and used on any system with Python, `uv`, and Ollama.

## Features

- **Self-contained:**  The entire agent is a single Python file.
- **HDF5 File Exploration:** Easily navigate the HDF5 file structure (groups, datasets).
- **Data Analysis:** Retrieve dataset information (shape, dtype) and read data slices.
- **Local LLM Support:** Uses Ollama for local LLM execution (default model: `phi4:latest`).
- **Dependency Management:** Uses `uv` for automatic dependency installation.
- **Interactive and CLI Modes:** Supports both interactive and command-line usage.

## Available Tools

The agent uses a tool-based architecture. The LLM can call the following tools:

1.  **list_files**: Lists HDF5 files in a directory.
2.  **list_groups**: Lists groups within an HDF5 file.
3.  **list_datasets**: Lists datasets within an HDF5 group. (Currently unimplemented)

## Requirements

- Python 3.8+
- `uv` (Universal Python Package Installer): [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- Ollama: [https://ollama.ai/](https://ollama.ai/)

## Installation

1.  **Install `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install and start Ollama:**

    Follow the instructions on the Ollama website: [https://ollama.ai/](https://ollama.ai/)

3. **Pull the `phi4:latest` model (or another model if you prefer):**
   ```bash
   ollama pull phi4:latest
   ```

4.  **Place `sfa_hdf5_ollama.py` in your data directory:**  Put the script in the directory containing your HDF5 files.

## Usage

The script can be run directly using `uv` run.  Dependencies are automatically installed based on the `# /// script` block at the beginning of the file.

**Basic Usage (Command-line):**

```bash
uv run sfa_hdf5_ollama.py <directory_path> "<your_query>"
```

-   `<directory_path>`:  The path to the directory containing your HDF5 files (and the `sfa_hdf5_ollama.py` script).
-   `<your_query>`: Your natural language query.

**Example:**

```bash
uv run sfa_hdf5_ollama.py data "What groups are in the HDF5 file?"
```

**Interactive Mode:**

If you run the script without a query, it will enter interactive mode:

```bash
uv run sfa_hdf5_ollama.py data
```

You'll be prompted to enter queries. Type `exit` or `quit` to exit.

**Command-line Arguments:**

-   `directory`: (Required) The directory containing HDF5 files.
-   `query`: (Optional) The query to process. If not provided, the script enters interactive mode.
-    `-m`, `--model`: (Optional)  Specify a different Ollama model. Default is `phi4:latest`.

**Example (using a different model):**

```bash
uv run sfa_hdf5_ollama.py -m mistral data "What groups are in the HDF5 file?"
```

## Dependencies

The script's dependencies are listed in the `# /// script` block at the beginning of `sfa_hdf5_ollama.py`.  They will be automatically installed by `uv` when you run the script.  The dependencies are:

```
h5py>=3.8.0,<3.9.0
requests~=2.31.0
rich~=13.3.5
pydantic>=2.4.2,<3.0.0
ollama~=0.1.6
psutil~=5.9.5
numpy>=1.24.0,<2.0.0
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License.
