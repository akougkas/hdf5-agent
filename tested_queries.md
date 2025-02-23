# Test Queries

## Basic Operations

### List Available Files
```bash
uv run sfa_hdf5_ollama.py data "List all HDF5 files in the directory."
```
Purpose: Tests the optimized list_files tool with async globbing, ensuring quick file discovery.

### Basic Group Exploration
```bash
uv run sfa_hdf5_ollama.py data "What groups are in test_data.h5?"
```
Purpose: Uses cached file access in list_groups for faster group listing.

### Nested Group Listing
```bash
uv run sfa_hdf5_ollama.py data "Show all groups in test_data.h5."
```
Purpose: Tests efficient group iteration with minimal I/O via cached h5py.File.

## Dataset Operations

### Dataset Discovery in Root
```bash
uv run sfa_hdf5_ollama.py data "List all datasets in the root group of test_data.h5."
```
Purpose: Tests list_datasets with the root group (/) using optimized iteration.

### Specific Group Datasets
```bash
uv run sfa_hdf5_ollama.py data "What datasets are in the /measurements group of test_data.h5?"
```
Purpose: Verifies dataset listing in a specific group with cached file access.

## Complex Queries

### Multi-Step Exploration
```bash
uv run sfa_hdf5_ollama.py data "List groups in test_data.h5 and then list datasets in /measurements."
```
Purpose: Tests concurrent tool execution (list_groups and list_datasets) with a single LLM call, leveraging the new processing agent.

### Conditional Exploration
```bash
uv run sfa_hdf5_ollama.py data "Find all groups in test_data.h5 with datasets and summarize one."
```
Purpose: Combines list_datasets and summarize_dataset in parallel, testing multi-tool efficiency.

## Error Handling

### Invalid File
```bash
uv run sfa_hdf5_ollama.py data "Show contents of nonexistent.h5."
```
Purpose: Ensures proper error reporting with minimal LLM overhead.

### Invalid Group Path
```bash
uv run sfa_hdf5_ollama.py data "List datasets in /not/a/real/path in test_data.h5."
```
Purpose: Tests error handling for invalid group paths with cached file checks.

### Complex Error Case
```bash
uv run sfa_hdf5_ollama.py data "List groups in nonexistent.h5 then datasets in test_data.h5."
```
Purpose: Verifies graceful handling of mixed valid/invalid operations in a single query.

## Test Queries for read_dataset_data

### Full Dataset Read (Small Numerical Data)
```bash
uv run sfa_hdf5_ollama.py data "Read all data from 'numerical_data/integers_1d' in test_data.h5."
```
Purpose: Tests full dataset reading with optimized slicing and async I/O.

### Slicing First N Elements (Timeseries)
```bash
uv run sfa_hdf5_ollama.py data "Read the first 10 elements of 'timeseries/temperature' in test_data.h5."
```
Purpose: Validates slicing with dynamic MAX_SLICE_SIZE enforcement.

### Specific Range Slicing (Numerical 2D)
```bash
uv run sfa_hdf5_ollama.py data "Read rows 2 to 5 and columns 1 to 3 of 'numerical_data/floats_2d' in test_data.h5."
```
Purpose: Tests 2D slicing with efficient async data retrieval.

### Large Dataset Size Limit Test (Compressed 3D)
```bash
uv run sfa_hdf5_ollama.py data "Read all data from 'numerical_data/compressed_3d' in test_data.h5."
```
Purpose: Ensures MAX_SLICE_SIZE limits are respected for large datasets.

### Text Data Read (Full)
```bash
uv run sfa_hdf5_ollama.py data "Read all data from 'text_data/fruits' in test_data.h5."
```
Purpose: Tests text data handling with optimized numpy string conversion.

### Nested Dataset Slicing
```bash
uv run sfa_hdf5_ollama.py data "Read the first 5 elements of 'nested/level1/level2/level3/data_3' in test_data.h5."
```
Purpose: Verifies deep path handling with cached file access.

### Non-Existent Dataset Error
```bash
uv run sfa_hdf5_ollama.py data "Read 'numerical_data/fake_data' in test_data.h5."
```
Purpose: Tests error handling for missing datasets.

### Text Data Comparison
```bash
uv run sfa_hdf5_ollama.py data "What datasets exist in 'text_data/' in test_data.h5 and how do they compare to 'text_data/fruits'?"
```
Purpose: Combines list_datasets and compare_datasets in a single optimized run, leveraging parallelism.

## Notes on Usage
Assumptions: These queries assume test_data.h5 contains the same structure as in your original tests (e.g., text_data/fruits, numerical_data/integers_1d, etc.). Adjust paths if your file differs.
