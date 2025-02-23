# Key Optimization Opportunities

## 1. HDF5 I/O Optimization (h5py)
**Problem:** Repeated file opening/closing in every tool function is inefficient, especially for large files. Each `h5py.File()` call involves disk I/O and locking overhead.

**Solution:** Use a context manager or a singleton-like HDF5 file cache to keep files open during a query session, reducing I/O overhead. Leverage h5py's chunking and caching capabilities for large datasets.

**Impact:** Significant speed-up for multi-step queries on the same file.

## 2. LLM Interaction Efficiency
**Problem:** The LLM is pinged repeatedly in a loop (up to 10 iterations), often with large system prompts, leading to high token usage and latency.

**Solution:**
- Minimize system prompt size by pre-parsing queries into tool calls when possible, reducing LLM reliance
- Use a single LLM call with structured output (e.g., a list of tool calls) instead of iterative back-and-forth
- Compress accumulated results in messages to avoid token bloat

**Impact:** Fewer LLM calls, lower token count, faster response times.

## 3. Serial Pipeline Bottlenecks
**Problem:** Tools are executed sequentially within an async loop, even though some operations (e.g., summarizing multiple datasets) could be parallelized.

**Solution:** Use `asyncio.gather()` to run independent tool calls concurrently when the query allows.

**Impact:** Reduced wall-clock time for multi-tool queries.

## 4. Memory Management
**Problem:** Large dataset reads (e.g., `read_dataset_data`) load entire slices into memory without checking available RAM, risking crashes with big files.

**Solution:** Stream data in chunks using HDF5's chunked I/O and process incrementally. Adjust `MAX_SLICE_SIZE` dynamically based on system resources.

**Impact:** Better scalability for large HDF5 files.

## 5. Profiling and Metrics
**Problem:** Metrics tracking (e.g., `METRICS`) is coarse-grained and adds overhead with redundant dictionary updates.

**Solution:** Use `contextlib` with a lightweight timing decorator and aggregate metrics only at the end, reducing runtime overhead.

**Impact:** Cleaner, faster profiling with minimal performance hit.

## 6. System Calls and Redundant Code
**Problem:** Repeated `os.path.exists()` checks and redundant path normalization in every tool function waste CPU cycles.

**Solution:** Centralize file existence checks and path handling in a single utility function or class.

**Impact:** Fewer system calls, cleaner code.

## 7. Tool Function Efficiency
**Problem:** Functions like `list_groups` and `list_datasets` use `visit()` or `visititems()`, which can be slow for deep hierarchies, and lack caching for repeated calls.

**Solution:** Use HDF5's native iteration methods (`keys()` or `__iter__`) with a custom cache for frequently accessed metadata.

**Impact:** Faster traversal and reduced I/O for common operations.

## 8. Asynchronous Overhead
**Problem:** Async is used, but many operations (e.g., HDF5 reads) are synchronous, negating concurrency benefits.

**Solution:** Offload I/O-bound HDF5 operations to a thread pool using `asyncio.to_thread()` for true parallelism.

**Impact:** Better utilization of async framework.

# Extended Functionality (Lower Priority)

## HDF5 File Creation/Modification
- **Functionality:** Allow the agent to create new HDF5 files or modify existing ones (add datasets, groups, attributes).
- **Implementation Details:**
    - This would require new tools and careful consideration of safety and permissions
    - The agent should not be allowed to overwrite existing files without explicit confirmation
- **Test Queries:**
    ```python
    # These are examples, and the exact syntax would depend on the tool design
    "Create a new HDF5 file named 'new_data.h5'."
    "Add a dataset named 'pressure' to 'new_data.h5' with shape (100,) and dtype float32."
    "Set the attribute 'units' to 'Pa' for the 'pressure' dataset in 'new_data.h5'."
    ```
