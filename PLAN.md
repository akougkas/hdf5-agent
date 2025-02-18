# Improvement Plan for HDF5 Single File Agent

### Areas for Improvement

1. **Code Organization and Documentation**
   ```python:sfa_hdf5_ollama.py
   startLine: 1
   endLine: 50
   ```
   - Add module-level docstring
   - Group constants at the top (OLLAMA_MODEL is missing)
   - Add type hints to all function return values

2. **Performance Optimization**
   ```python:sfa_hdf5_ollama.py
   startLine: 274
   endLine: 313
   ```
   - Add prompt caching for frequently accessed metadata
   - Add progress indicators for long operations


## Improvement Plan

### 1. Code Organization
```python
# Add to top of file
"""
HDF5 Single File Agent

A powerful agent for exploring and analyzing HDF5 files through natural language queries.
Built on top of Ollama for local LLM execution.

Author: [Your Name]
License: MIT
"""

# Constants
OLLAMA_MODEL = "phi4"
OLLAMA_ENDPOINT = "http://localhost:11434"
MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 1024  # 1MB
```

### 2. Enhanced Error Handling
```python
from typing import Optional, Union
from dataclasses import dataclass
import logging

@dataclass
class OperationResult:
    success: bool
    data: Optional[Union[str, dict]] = None
    error: Optional[str] = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### 3. Performance Improvements
```python
from functools import lru_cache
from tqdm import tqdm

@lru_cache(maxsize=100)
def get_dataset_metadata(file_path: str, dataset_path: str) -> dict:
    """Cache metadata for frequently accessed datasets."""
    # Implementation
```

### 4. Testing Framework
```python
# tests/test_hdf5_agent.py
import pytest
from pathlib import Path
import h5py
import numpy as np

@pytest.fixture
def sample_hdf5_file(tmp_path):
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, 'w') as f:
        f.create_dataset("data", data=np.random.rand(100, 100))
    return file_path
```

## Implementation Steps

1. **Phase 1: Core Improvements**
   - Add proper logging
   - Implement constants and configuration
   - Add caching for metadata
   - Add progress bars for long operations

2. **Phase 2: Testing**
   - Set up pytest framework
   - Write unit tests for each tool
   - Add integration tests
   - Add test data generation

3. **Phase 3: Documentation**
   - Add detailed docstrings
   - Create API documentation
   - Add usage examples
   - Create contribution guidelines

4. **Phase 4: Performance**
   - Implement chunked reading
   - Add parallel processing for large datasets
   - Optimize memory usage
   - Add performance metrics

## Running Instructions

1. Install additional dependencies:
```bash
uv pip install pytest tqdm rich[logging] pytest-cov
```

2. Set up environment:
```bash
export OLLAMA_MODEL=phi4
export OLLAMA_ENDPOINT=http://localhost:11434
```

3. Run tests:
```bash
pytest tests/ -v --cov=sfa_hdf5_ollama
```

4. Run the agent:
```bash
python sfa_hdf5_ollama.py /path/to/data "What datasets are available?"
```

## Future Enhancements

1. **Visualization Support**
   - Add hdf5_view integration
   - Support for basic plotting commands
   - Interactive visualization options

2. **Extended Functionality**
   - Add support for HDF5 file creation/modification
   - Implement dataset comparison tools
   - Add support for complex queries

3. **User Experience**
   - Add interactive mode
   - Implement command history
   - Add auto-completion
   - Add progress visualization

4. **Performance Monitoring**
   - Add telemetry
   - Implement performance logging
   - Add resource usage monitoring

## Agent Testing Commands

### Basic Operations
1. List available files:
```bash
uv run sfa_hdf5_ollama.py data "List all HDF5 files in this directory"
```

2. Basic group exploration:
```bash
uv run sfa_hdf5_ollama.py data "What groups are in test_data.h5?"
```

3. Nested group listing:
```bash
uv run sfa_hdf5_ollama.py data "Show me all groups and their subgroups in test_data.h5"
```

### Dataset Operations
4. Dataset discovery:
```bash
uv run sfa_hdf5_ollama.py data "List all datasets in the root group of test_data.h5"
```

5. Specific group datasets:
```bash
uv run sfa_hdf5_ollama.py data "What datasets are available in the /measurements group of test_data.h5?"
```

### Complex Queries
6. Multi-step exploration:
```bash
uv run sfa_hdf5_ollama.py data "First show me all groups in test_data.h5, then list the datasets in the deepest group"
```

7. Conditional exploration:
```bash
uv run sfa_hdf5_ollama.py data "Find all groups in test_data.h5 that contain datasets"
```

### Error Handling
8. Invalid file:
```bash
uv run sfa_hdf5_ollama.py data "Show me the contents of nonexistent.h5"
```

9. Invalid group path:
```bash
uv run sfa_hdf5_ollama.py data "List datasets in /not/a/real/path within test_data.h5"
```

10. Complex error case:
```bash
uv run sfa_hdf5_ollama.py data "First list groups in nonexistent.h5, then show datasets in test_data.h5"
```

These test cases progressively evaluate:
- Basic file and group listing
- Nested group navigation
- Dataset discovery
- Multi-step operations
- Error handling
- Complex query processing
- Response formatting
- Tool selection logic

Results from these tests will inform the prioritization of improvements in Phase 1 implementation.
