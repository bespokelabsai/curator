"""Build script for packaging the curator viewer application.

This script handles the build process for the curator viewer, including npm installation,
Next.js build compilation, and running tests. It manages copying build artifacts and
handles file exclusions during the copy process.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """Execute a shell command in the specified directory.

    Args:
        command: The shell command to execute
        cwd: The working directory to run the command in (optional)

    Returns:
        subprocess.CompletedProcess: The result of the command execution
    """
    result = subprocess.run(command, shell=True, cwd=cwd, check=True)
    return result


def npm_install():
    """Install npm dependencies for the bespoke-dataset-viewer."""
    print("Running npm install")
    run_command("npm install", cwd="bespoke-dataset-viewer")


def copy_with_excludes(source, target, excludes=None):
    """Copy files/directories while excluding specified paths.

    Args:
        source: Source path to copy from
        target: Target path to copy to
        excludes: List of paths to exclude from copying (optional)
    """
    if excludes is None:
        excludes = []

    if source.is_file():
        shutil.copy2(source, target)
        print(f"Copied file {source} to {target}")
    elif source.is_dir():
        if target.exists():
            shutil.rmtree(target)

        def ignore_patterns(path, names):
            return [n for n in names if str(Path(path) / n) in excludes]

        shutil.copytree(source, target, ignore=ignore_patterns)
        print(f"Copied directory {source} to {target}")


def nextjs_build():
    """Build the Next.js application and copy build artifacts.

    Runs the Next.js build process and copies the resulting files to the static folder,
    excluding specified paths like the Next.js cache directory.
    """
    print("Running Next.js build")
    run_command("npm run build", cwd="bespoke-dataset-viewer")
    print("Copying build artifacts to static folder")

    # Source and target directories
    source_base = Path("bespoke-dataset-viewer")
    target_base = Path("src/bespokelabs/curator/viewer/static")

    # Ensure target directory exists
    if target_base.exists():
        shutil.rmtree(target_base)
    target_base.mkdir(parents=True, exist_ok=True)

    # Files and directories to copy
    files_to_copy = [
        ".next",
        "app",
        "components",
        "lib",
        "public",
        "types",
        "package.json",
        "package-lock.json",
        "next.config.ts",
        "next-env.d.ts",
        "tsconfig.json",
        "postcss.config.mjs",
        "tailwind.config.ts",
        "components.json",
    ]

    # Paths to exclude
    exclude_paths = [str(source_base / ".next" / "cache")]

    for item in files_to_copy:
        source = source_base / item
        target = target_base / item

        if source.exists():
            copy_with_excludes(source, target, exclude_paths)
        else:
            print(f"Warning: {source} not found")


def run_pytest():
    """Run pytest and exit if tests fail."""
    print("Running pytest")
    try:
        run_command("pytest -v")
    except subprocess.CalledProcessError:
        print("Pytest failed. Aborting build.")
        sys.exit(1)


def main():
    """Execute the full build process.

    Runs npm install, builds the Next.js application, and runs tests.
    """
    npm_install()
    nextjs_build()
    run_pytest()
    print("Build completed successfully.")


if __name__ == "__main__":
    main()
