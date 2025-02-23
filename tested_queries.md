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




### Test Queries for read_dataset_data

#### Full Dataset Read (Small Numerical Data)
- **Test Name**: ReadFullIntegers1D
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read all data from dataset 'numerical_data/integers_1d' in 'test_data.h5'."
```
- **Purpose**: Tests reading an entire 1D dataset, assuming it's small enough to fit under MAX_SLICE_SIZE (1M elements).

#### Slicing First N Elements (Timeseries)
- **Test Name**: ReadFirst10Temperature
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read the first 10 elements of dataset 'timeseries/temperature' in 'test_data.h5'."
```
- **Purpose**: Tests slicing a 1D dataset with a simple range (0 to 10).

#### Specific Range Slicing (Numerical 2D)
- **Test Name**: ReadFloats2DSlice
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read rows 2 to 5 and columns 1 to 3 of dataset 'numerical_data/floats_2d' in 'test_data.h5'."
```
- **Purpose**: Tests 2D slicing with specific row and column ranges, assuming floats_2d is a 2D array.

#### Large Dataset Size Limit Test (Compressed 3D)
- **Test Name**: ReadLargeCompressed3D
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read all data from dataset 'numerical_data/compressed_3d' in 'test_data.h5'."
```
- **Purpose**: Tests the MAX_SLICE_SIZE limit, assuming compressed_3d might exceed 1M elements due to its 3D nature.

#### Text Data Read (Full)
- **Test Name**: ReadTextFruits
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read all data from dataset 'text_data/fruits' in 'test_data.h5'."
```
- **Purpose**: Tests reading a text-based dataset, assuming it's small enough to fit under MAX_SLICE_SIZE.

#### Nested Dataset Slicing
- **Test Name**: ReadNestedLevel3Slice
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read the first 5 elements of dataset 'nested/level1/level2/level3/data_3' in 'test_data.h5'."
```
- **Purpose**: Tests reading from a deeply nested dataset with a small slice.

#### Non-Existent Dataset Error
- **Test Name**: ReadNonExistentDataset
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read dataset 'numerical_data/fake_data' in 'test_data.h5'."
```
- **Purpose**: Tests error handling for a dataset that doesn't exist.

#### Image Data Partial Read
- **Test Name**: ReadImageNoiseSlice
- **Test CLI Command**:
```bash
uv run sfa_hdf5_ollama.py data "Read rows 0 to 10 and columns 0 to 10 of dataset 'images/noise' in 'test_data.h5'."
```
- **Purpose**: Tests slicing a 2D image dataset, assuming noise is a 2D array (e.g., a small image).


```bash
uv run sfa_hdf5_ollama.py data "what other datasets do exist in 'text_data/' in 'test_data.h5' and how do they compare to 'text_data/fruits?"
```
