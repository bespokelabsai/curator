"""Test script for installation UI."""

import argparse
import importlib.util
import os
import sys


def import_install_ui():
    """Import just the install_ui module without importing the whole package."""
    # Get the absolute path to install_ui.py
    install_ui_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # Go up one level since we're in tests/
        "src/bespokelabs/curator/install_ui.py",
    )

    # Import the module directly from file
    spec = importlib.util.spec_from_file_location("install_ui", install_ui_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["install_ui"] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Run the test script with command line arguments."""
    parser = argparse.ArgumentParser(description="Test the installation UI.")
    parser.add_argument(
        "--scenario",
        choices=["success", "error"],
        default="success",
        help="Which scenario to test (success or error)",
    )
    args = parser.parse_args()

    # Import just the install_ui module
    install_ui = import_install_ui()

    # Run the enhanced install based on scenario
    if args.scenario == "success":
        install_ui.enhanced_install("bespokelabs-curator")
    else:
        install_ui.enhanced_install("nonexistent-package-12345")


if __name__ == "__main__":
    main()
