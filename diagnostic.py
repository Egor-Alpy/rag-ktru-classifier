#!/usr/bin/env python3
"""
Diagnostic script to check environment and imports.
Place this in each service directory and run it to diagnose import issues.
"""

import sys
import os
import importlib


def check_imports(module_names):
    """Try to import each module and report success/failure."""
    results = {}
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            results[module_name] = "Success"
        except ImportError as e:
            results[module_name] = f"Failed: {str(e)}"
    return results


def main():
    """Main diagnostic function."""
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

    print("\n=== Directory Structure ===")
    for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
        level = root.count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

    # Common modules to check
    modules_to_check = [
        'pydantic',
        'pydantic_settings',
        'fastapi',
        'uvicorn',
        'app',
        'app.main',
        'app.config'
    ]

    print("\n=== Import Tests ===")
    results = check_imports(modules_to_check)
    for module, result in results.items():
        print(f"{module}: {result}")


if __name__ == "__main__":
    main()