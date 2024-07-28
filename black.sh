#!/bin/bash

# Check if the correct number of arguments are passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the argument
DIRECTORY=$1

# Find all .py files in the directory recursively and format them with black
find "$DIRECTORY" -type f -name "*.py" -exec black {} +

# Check if black formatting was successful
if [ $? -eq 0 ]; then
    echo "All Python files in $DIRECTORY have been formatted with black."
else
    echo "An error occurred while formatting Python files."
    exit 1
fi