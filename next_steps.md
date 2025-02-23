**Extended Functionality (Lower Priority)**

1.  **HDF5 File Creation/Modification:**
    *   **Functionality:**  Allow the agent to create new HDF5 files or modify existing ones (add datasets, groups, attributes).
    *   **Implementation Details:**
        *   This would require new tools and careful consideration of safety and permissions.  The agent should not be allowed to overwrite existing files without explicit confirmation.
    *   **Test Queries:** (These are examples, and the exact syntax would depend on the tool design)
        *   "Create a new HDF5 file named 'new_data.h5'."
        *   "Add a dataset named 'pressure' to 'new_data.h5' with shape (100,) and dtype float32."
        *   "Set the attribute 'units' to 'Pa' for the 'pressure' dataset in 'new_data.h5'."
