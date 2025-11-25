import unittest
import os
import sys
import pandas as pd
import numpy as np

# Add backend to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tool_registry import get_tools_map, generate_tool_documentation
from code_executor import execute_python_code

class TestToolRegistry(unittest.TestCase):
    
    def test_get_tools_map(self):
        """Verify that the tools map returns the expected functions."""
        tools = get_tools_map()
        self.assertIn("get_schism_node_data", tools)
        self.assertIn("calculate_wave_velocity", tools)
        self.assertIn("get_elevation_difference", tools)
        self.assertIn("filter_by_time_window", tools)
        self.assertIn("filter_by_point", tools)
        self.assertIn("filter_by_bbox", tools)
        self.assertIn("get_dataset_summary", tools)
        
        # Verify they are callables
        self.assertTrue(callable(tools["get_schism_node_data"]))

    def test_generate_tool_documentation(self):
        """Verify that documentation string is generated correctly."""
        doc_str = generate_tool_documentation()
        self.assertIn("### 🛠️ AVAILABLE DOMAIN TOOLS:", doc_str)
        self.assertIn("get_schism_node_data", doc_str)
        self.assertIn("calculate_wave_velocity", doc_str)
        self.assertIn("filter_by_point", doc_str)

    def test_executor_integration(self):
        """
        Verify that the executor can call the injected tools.
        We won't load a real NetCDF here to avoid large file dependencies in unit tests,
        but we will test a tool that doesn't require external files, like calculate_wave_velocity,
        or we can mock the data loading if needed.
        
        Actually, let's test `calculate_wave_velocity` by creating a dataframe inside the executor code.
        """
        
        # Python code to run inside the executor
        # It creates a dummy dataframe and calls the tool
        code_to_run = """
import pandas as pd
import numpy as np

# Create dummy data
df = pd.DataFrame({
    'wsh_x': [3.0, 4.0],
    'wsh_y': [4.0, 3.0]
})

# Call the injected tool
df_result = calculate_wave_velocity(df)

print(f"Result: {df_result['wave_velocity'].tolist()}")
"""
        
        # Use a real file so xr.open_dataset doesn't fail
        real_nc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_velocity_raster.nc")
        
        result = execute_python_code(code_to_run, netcdf_path=real_nc_path)
        
        if not result["success"]:
            print(f"Executor failed with stderr: {result['stderr']}")
        
        self.assertTrue(result["success"])
        self.assertIn("Result: [5.0, 5.0]", result["stdout"])
        # We don't strictly check stderr because of potential RuntimeWarnings from numpy/libraries
        if result["stderr"]:
            print(f"Note: Stderr contained: {result['stderr']}")

if __name__ == "__main__":
    unittest.main()
