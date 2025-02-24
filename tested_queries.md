## All Commands
```bash
# Basic Operations
uv run sfa_hdf5_ollama.py data "List HDF5 files in the directory."
uv run sfa_hdf5_ollama.py data "What groups are in test_data.h5?"

# Dataset Operations
uv run sfa_hdf5_ollama.py data "List all datasets in the 'test_data.h5.' file"
uv run sfa_hdf5_ollama.py data "What datasets are in the /images group of test_data.h5?"

# Complex Queries
uv run sfa_hdf5_ollama.py data "List groups in test_data.h5 and then list datasets in /images."
uv run sfa_hdf5_ollama.py data "Find all groups in test_data.h5 with datasets and summarize one of them that seems interesting."
# Error Handling 
uv run sfa_hdf5_ollama.py data "Show contents of nonexistent.h5."
uv run sfa_hdf5_ollama.py data "List datasets in /not/a/real/path in test_data.h5."
uv run sfa_hdf5_ollama.py data "List groups in nonexistent.h5 then datasets in test_data.h5."

# Dataset Summary
uv run sfa_hdf5_ollama.py data "Summarize dataset 'timeseries/temperature' inside the timeseries group in the test_data.h5."
