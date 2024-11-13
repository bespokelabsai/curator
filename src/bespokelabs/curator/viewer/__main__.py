import os
import subprocess
import sys
from pathlib import Path
from argparse import ArgumentParser
import webbrowser
from contextlib import closing
import socket
import logging
import time
import platform
import tempfile
import shutil


def get_viewer_path():
    return str(Path(__file__).parent)


def ensure_dependencies():
    """Ensure npm dependencies are installed"""
    static_dir = os.path.join(get_viewer_path(), "static")
    node_modules = os.path.join(static_dir, "node_modules")

    if not os.path.exists(node_modules):
        print("First run: Installing Node.js dependencies...")
        print("Your node_modules path: ", node_modules)
        try:
            subprocess.run(["npm", "install"], cwd=static_dir, check=True)
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(
                "Error: Node.js is not installed. Please install Node.js to run the viewer."
            )
            sys.exit(1)


def _setup_logging(level):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s] %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_node_installed():
    """Check if Node.js is installed and return version if found"""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    parser = ArgumentParser(description="Curator Viewer")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the server on (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to run the server on (default: 3000)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enables debug logging for more verbose output",
    )
    args = parser.parse_args()

    _setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Check if Node.js is installed
    node_version = check_node_installed()
    if not node_version:
        print("\nNode.js is required but not found on your system.")
        print("Please install Node.js from https://nodejs.org/")
        print("\nAfter installing:")
        print("1. Close and reopen your terminal")
        print("2. Verify installation by running: node --version")
        print("3. Run curator-viewer again")
        sys.exit(1)

    ensure_dependencies()

    # Set environment variables for the Next.js server
    env = os.environ.copy()
    env["NODE_ENV"] = "production"
    env["HOST"] = args.host
    env["PORT"] = str(args.port)

    # Start the Next.js server
    viewer_path = get_viewer_path()
    static_dir = os.path.join(viewer_path, "static")
    server_file = os.path.join(viewer_path, "server.js")

    if not os.path.exists(os.path.join(static_dir, ".next")):
        print(
            "Error: Next.js build artifacts not found. The package may not be built correctly."
        )
        sys.exit(1)

    try:
        subprocess.run(
            ["node", server_file], cwd=viewer_path, env=env, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error starting Next.js server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: Node.js is not installed. Please install Node.js to run the viewer."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
