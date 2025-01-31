"""Utility functions for launching and configuring experiments and ray tuning."""

import importlib.util
import os


def import_class_from_file(file_path: str) -> type:
    """Import and return the Model class from a specified Python file.

    Args:
        file_path (str): Path to the Python file containing the Model class.

    Returns:
        type: The Model class found in the file.

    Raises:
        ImportError: If no class starting with 'Model' is found in the file.
    """
    # Extract directory path and file name
    directory, file_name = os.path.split(file_path)
    module_name = os.path.splitext(file_name)[0]  # Remove extension to get module name

    # Create a module from the file path
    # In summary, these three lines of code are responsible for creating a module specification based on a file location, creating a module object from that specification, and then executing the module's code to populate the module object with the definitions from the Python file.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {file_path}")
    spec.loader.exec_module(module)

    # Find the class dynamically
    for name in dir(module):
        model_class = getattr(module, name)
        if isinstance(model_class, type) and name.startswith("Model"):
            return model_class

    # Class not found
    raise ImportError("No class starting with 'Model' found in the file.")
