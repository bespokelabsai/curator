#!/bin/bash

# Install 
poetry install --extras "code_execution vllm" --with dev

# Build Curator Viewer
python build_pkg.py

# Build Curator Package
poetry build

# Publish Curator Package
poetry publish

# Clean up
rm -rf src/bespokelabs/curator/viewer/static
rm -rf dist
