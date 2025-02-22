**Core Data Access and Summarization**

1.  **Read Dataset Data:**
    *   **Functionality:** Read data from a specified dataset within an HDF5 file, handling slicing and respecting `MAX_SLICE_SIZE` to prevent memory issues.  The function should correctly handle various HDF5 data types and return a string representation of the data.
    *   **Implementation Details:**
        *   Create a Pydantic model for input parameters (`file_path`, `dataset_path`, `slice_start`, `slice_end`).  `file_path` should be relative to the directory.
        *   Use `h5py` to open the file and dataset.
        *   Implement robust slicing logic, calculating the requested slice size and comparing it against `MAX_SLICE_SIZE` *before* attempting to read the data.  Return an error message if the size is exceeded.
        *   Convert the read data to a string representation.  Consider how to best represent different data types (numerical arrays, strings, etc.) in a way that's useful to the LLM.
        *   Handle potential `h5py.H5Error` and `KeyError` exceptions gracefully.
    *   **Test Queries:**
        *   "Read the first 10 elements of dataset 'data' in file 'test.h5'."
        *   "Read all data from dataset 'my_data' in 'results.h5'." (Test with a dataset small enough to fit within `MAX_SLICE_SIZE`)
        *   "Read elements 5 to 15 of the 'temperature' dataset in 'sensor_data.h5'."
        *   "Try to read a very large slice from dataset 'big_data' in 'large_file.h5'." (This should trigger the `MAX_SLICE_SIZE` error.)
        *   "Read dataset 'nonexistent_dataset' in 'test.h5'." (Test `KeyError` handling)
        *   "Read dataset 'data' from file 'corrupted.h5'." (Test `h5py.H5Error` handling, requires a corrupted file)
    *   **Update Agent Prompt:**
        *   The `AGENT_PROMPT` should be updated to reflect the new tools and their usage.
        *   Clearly describe the purpose of the new tool.
        *   Explain the required and optional parameters for the tool.
        *   Provide examples of how the tool should be used in JSON format.
        *   Emphasize the `MAX_SLICE_SIZE` restriction and its importance.

2.  **Summarize Dataset:**
    *   **Functionality:**  Provide a concise summary of a dataset's contents.  This goes beyond simply reading the raw data and involves some level of analysis.
    *   **Implementation Details:**
        *   Could be a separate tool or integrated into the `read_dataset` tool.
        *   For numerical data, calculate statistics like min, max, mean, median, standard deviation, and potentially quantiles.
        *   For string data, consider providing information about the number of unique values, the most frequent values, or a sample of values.
        *   Use NumPy for efficient numerical calculations.
        *   Return a string representation of the summary.
    *   **Test Queries:**
        *   "Summarize dataset 'data' in file 'test.h5'."
        *   "What are the statistics for the 'temperature' dataset in 'sensor_data.h5'?"
        *   "Describe the contents of dataset 'log_messages' in 'application.h5'."

3.  **Dataset Comparison:**
    *   **Functionality:**  Compare two datasets (potentially in different files) for equality or differences.
    *   **Implementation Details:**
        *   This would involve reading data from both datasets and using NumPy's comparison functions (e.g., `np.array_equal`, `np.allclose`).
        *   The tool should handle cases where the datasets have different shapes or data types.
    *   **Test Queries:**
        *   "Compare dataset 'data' in 'test.h5' with dataset 'data_backup' in 'backup.h5'."
        *   "Are the 'temperature' datasets in 'sensor1.h5' and 'sensor2.h5' identical?"

4. **Update Agent Prompt:**
    * **Functionality:** The `AGENT_PROMPT` should be updated to reflect the new tools and their usage.
    * **Implementation Details:**
        *   Clearly describe the purpose of each new tool (`read_dataset`, `get_dataset_info`, etc.).
        *   Explain the required and optional parameters for each tool.
        *   Provide examples of how the tools should be used in JSON format.
        *   Emphasize the `MAX_SLICE_SIZE` restriction and its importance.




**Extended Functionality (Lower Priority)**

1.  **HDF5 File Creation/Modification:**
    *   **Functionality:**  Allow the agent to create new HDF5 files or modify existing ones (add datasets, groups, attributes).
    *   **Implementation Details:**
        *   This would require new tools and careful consideration of safety and permissions.  The agent should not be allowed to overwrite existing files without explicit confirmation.
    *   **Test Queries:** (These are examples, and the exact syntax would depend on the tool design)
        *   "Create a new HDF5 file named 'new_data.h5'."
        *   "Add a dataset named 'pressure' to 'new_data.h5' with shape (100,) and dtype float32."
        *   "Set the attribute 'units' to 'Pa' for the 'pressure' dataset in 'new_data.h5'."
