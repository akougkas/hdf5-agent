# HDF5 Test Data Generator
#
# Creates a test HDF5 file with various features:
# - Multiple groups and nested groups
# - Different dataset types (integers, floats, strings)
# - Different dataset shapes (1D, 2D, 3D)
# - Attributes at file, group, and dataset levels
# - Compressed datasets
# - External links
# - Different data types and special values (NaN, Inf)

import h5py
import numpy as np
import os
from datetime import datetime

def create_test_data(output_dir: str = "data") -> str:
    # Create a test HDF5 file with various features for testing the agent.
    #
    # Args:
    #     output_dir: Directory where the HDF5 file will be created
    #
    # Returns:
    #     Path to the created HDF5 file
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main test file
    file_path = os.path.join(output_dir, "test_data.h5")
    
    with h5py.File(file_path, 'w') as f:
        # File-level attributes
        f.attrs['creation_date'] = str(datetime.now())
        f.attrs['description'] = 'Test HDF5 file for agent testing'
        f.attrs['version'] = '1.0'
        
        # 1. Numerical Data Group
        numerical = f.create_group('numerical_data')
        numerical.attrs['group_type'] = 'Numerical datasets'
        
        # 1D dataset with integers
        int_data = numerical.create_dataset(
            'integers_1d',
            data=np.arange(100),
            dtype='int32'
        )
        int_data.attrs['description'] = '1D integer array from 0 to 99'
        
        # 2D dataset with floats
        float_data = numerical.create_dataset(
            'floats_2d',
            data=np.random.rand(50, 50),
            dtype='float64'
        )
        float_data.attrs['description'] = '2D random float array'
        float_data.attrs['stats_mean'] = float(np.mean(float_data[:]))
        float_data.attrs['stats_std'] = float(np.std(float_data[:]))
        
        # 3D dataset with compression
        compressed_data = numerical.create_dataset(
            'compressed_3d',
            data=np.random.randn(20, 20, 20),
            compression='gzip',
            compression_opts=9
        )
        compressed_data.attrs['description'] = '3D compressed random normal distribution'
        
        # Special values dataset
        special_values = np.array([np.nan, np.inf, -np.inf, 0, 1, -1])
        numerical.create_dataset('special_values', data=special_values)
        
        # 2. Time Series Group
        timeseries = f.create_group('timeseries')
        timeseries.attrs['group_type'] = 'Time series data'
        
        # Temperature readings
        timestamps = np.arange('2024-01-01', '2024-02-01', dtype='datetime64[D]')
        temperatures = 20 + 5 * np.sin(np.arange(len(timestamps)) * 0.2) + np.random.randn(len(timestamps))
        
        temp_data = timeseries.create_dataset('temperature', data=temperatures)
        temp_data.attrs['unit'] = 'Celsius'
        temp_data.attrs['sampling_rate'] = '1 day'
        temp_data.attrs['start_date'] = str(timestamps[0])
        
        # 3. Text Data Group
        text = f.create_group('text_data')
        text.attrs['group_type'] = 'Text and categorical data'
        
        # String dataset
        string_data = ['apple', 'banana', 'cherry', 'date', 'elderberry']
        text.create_dataset('fruits', data=string_data)
        
        # Categorical data
        categories = np.array(['A', 'B', 'C', 'A', 'B', 'C', 'A'], dtype='S1')
        cat_dataset = text.create_dataset('categories', data=categories)
        cat_dataset.attrs['categories'] = ['A', 'B', 'C']
        
        # 4. Nested Groups
        nested = f.create_group('nested')
        nested.attrs['description'] = 'Demonstration of nested groups'
        
        level1 = nested.create_group('level1')
        level2 = level1.create_group('level2')
        level3 = level2.create_group('level3')
        
        # Add datasets at each level
        for i, group in enumerate([nested, level1, level2, level3]):
            group.create_dataset(
                f'data_{i}',
                data=np.random.rand(10, 10)
            )
        
        # 5. Image-like Data
        images = f.create_group('images')
        images.attrs['group_type'] = 'Image-like 2D arrays'
        
        # Create a simple gradient image
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        gradient = X * Y
        
        images.create_dataset('gradient', data=gradient)
        
        # Create a "noisy" image
        noise = np.random.rand(64, 64)
        images.create_dataset('noise', data=noise)
        
    return file_path

if __name__ == "__main__":
    # Get the directory this script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test data
    output_file = create_test_data(script_dir)
    print(f"Created test HDF5 file: {output_file}")
    
    # Print structure of the created file
    with h5py.File(output_file, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        print("\nFile structure:")
        f.visititems(print_structure)
