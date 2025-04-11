#!/bin/bash

# Install 
make install

# Build Curator Package
poetry build

# Publish Curator Package
poetry publish
