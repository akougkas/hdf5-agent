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
uv run grok_sfa_hdf5_ollama.py data "First show me all groups in test_data.h5, then list the datasets in the deepest group"
```

7. Conditional exploration:
```bash
uv run grok_sfa_hdf5_ollama.py data "Find all groups in test_data.h5 that contain datasets"
```

### Error Handling
8. Invalid file:
```bash
uv run grok_sfa_hdf5_ollama.py data "Show me the contents of nonexistent.h5"
```

9. Invalid group path:
```bash
uv run grok_sfa_hdf5_ollama.py data "List datasets in /not/a/real/path within test_data.h5"
```

10. Complex error case:
```bash
uv run grok_sfa_hdf5_ollama.py data "First list groups in nonexistent.h5, then show datasets in test_data.h5"
```

