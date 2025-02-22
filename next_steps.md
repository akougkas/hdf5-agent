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

2.  **Summarize Dataset (Optional, but highly desirable):**
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

**Metadata Retrieval and Caching**

3.  **Get Dataset Information:**
    *   **Functionality:** Retrieve metadata about a specific dataset, including its shape, data type, and attributes.
    *   **Implementation Details:**
        *   Use `h5py`'s API to access `dataset.shape`, `dataset.dtype`, and `dataset.attrs`.
        *   Return a dictionary containing the metadata.
        *   Apply `@lru_cache` for caching.
    *   **Test Queries:**
        *   "What is the shape and data type of dataset 'data' in 'test.h5'?"
        *   "Get information about dataset 'my_data' in 'results.h5'."
        *   "Show me the attributes of the 'temperature' dataset in 'sensor_data.h5'."

4.  **Get Group Attribute:**
    *   **Functionality:** Retrieve the value of a specific attribute of a group.
    *   **Implementation Details:**
        *   Use `h5py` to access the group and its attributes (`group.attrs`).
        *   Return the attribute value.
    *   **Test Queries:**
        *   "What is the 'description' attribute of group '/metadata' in 'test.h5'?"
        *   "Get the 'version' attribute of group '/results/run1' in 'experiment.h5'."

5.  **Get File Attribute:**
    *   **Functionality:** Retrieve the value of a specific attribute of the HDF5 file itself.
    *   **Implementation Details:**
        *   Use `h5py` to access the file's attributes (`file.attrs`).
        *   Return the attribute value.
    *   **Test Queries:**
        *   "What is the 'creation_date' attribute of file 'test.h5'?"
        *   "Get the 'software_version' attribute of 'results.h5'."

6.  **Caching Implementation:**
    *   **Functionality:**  Cache the results of metadata retrieval functions (`get_dataset_info`, `get_dataset_attribute`, `get_group_attribute`, `get_file_attribute`) to improve performance.
    *   **Implementation Details:**
        *   Use `@lru_cache(maxsize=CACHE_SIZE, typed=False)` on each of the metadata functions.  `CACHE_SIZE` should be a configurable parameter.
    *   **Test Queries:**  (These are not direct queries, but rather tests of the caching mechanism)
        *   Call `get_dataset_info` for the same dataset multiple times and measure the execution time.  The second and subsequent calls should be significantly faster.
        *   Call `get_dataset_info` for `CACHE_SIZE + 1` different datasets.  The first dataset accessed should no longer be in the cache.

**Extended Functionality (Lower Priority)**

7.  **HDF5 File Creation/Modification:**
    *   **Functionality:**  Allow the agent to create new HDF5 files or modify existing ones (add datasets, groups, attributes).
    *   **Implementation Details:**
        *   This would require new tools and careful consideration of safety and permissions.  The agent should not be allowed to overwrite existing files without explicit confirmation.
    *   **Test Queries:** (These are examples, and the exact syntax would depend on the tool design)
        *   "Create a new HDF5 file named 'new_data.h5'."
        *   "Add a dataset named 'pressure' to 'new_data.h5' with shape (100,) and dtype float32."
        *   "Set the attribute 'units' to 'Pa' for the 'pressure' dataset in 'new_data.h5'."

8.  **Dataset Comparison:**
    *   **Functionality:**  Compare two datasets (potentially in different files) for equality or differences.
    *   **Implementation Details:**
        *   This would involve reading data from both datasets and using NumPy's comparison functions (e.g., `np.array_equal`, `np.allclose`).
        *   The tool should handle cases where the datasets have different shapes or data types.
    *   **Test Queries:**
        *   "Compare dataset 'data' in 'test.h5' with dataset 'data_backup' in 'backup.h5'."
        *   "Are the 'temperature' datasets in 'sensor1.h5' and 'sensor2.h5' identical?"

**User Experience and Monitoring**

9.  **Command History:**
    *   **Functionality:**  Implement a command history feature, allowing the user to recall and re-execute previous queries.
    *   **Implementation Details:** Store the history in a list or a file.

10. **Progress Indicator:**
    *   **Functionality:** Display a progress bar during long-running operations (e.g., reading large datasets).
    *   **Implementation Details:** Use the `rich.progress` library. Wrap the `h5py` data reading operations within `progress.track`.

11. **Detailed Performance Metrics:**
    *   **Functionality:** Track and report detailed performance metrics.
    *   **Implementation Details:**
        *   Use `time.perf_counter()` to measure the time spent in each tool function.
        *   Track the number of calls to each tool.
        *   Calculate the average dataset read size.
        *   Record the number of retries to Ollama.
        *   Store the metrics in a dictionary (e.g., `METRICS`).

12. **Update Agent Prompt:**
    * **Functionality:** The `AGENT_PROMPT` should be updated to reflect the new tools and their usage.
    * **Implementation Details:**
        *   Clearly describe the purpose of each new tool (`read_dataset`, `get_dataset_info`, etc.).
        *   Explain the required and optional parameters for each tool.
        *   Provide examples of how the tools should be used in JSON format.
        *   Emphasize the `MAX_SLICE_SIZE` restriction and its importance.
