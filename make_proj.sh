#!/bin/bash

if [ $# -eq 0 ] || [ $1 == "-h" ] || [ $1 == "--help" ]
then
  echo "Usage: $0 project_name"
  echo "Creates a new Python project with the specified name"
  exit 1
fi

PROJECT_NAME=$1

mkdir $PROJECT_NAME
cd $PROJECT_NAME

# Create files
touch config.yaml
touch driver.py

cat > README.md << EOF
# $PROJECT_NAME

## Introduction
Enter a brief introduction about your project here.

## Getting Started
Instructions on how to get started with your project.

## Instructions
Additional instructions on how to use your project.

## Theory
This section can be used to discuss theoretical aspects of your project.
EOF

mkdir $PROJECT_NAME
touch $PROJECT_NAME/__init__.py
touch $PROJECT_NAME/params.py

mkdir tests
touch tests/__init__.py

cat > tests/test_generic.py << EOF
import unittest

class TestGeneric(unittest.TestCase):
    def test_generic(self):
        assert True

if __name__ == '__main__':
    unittest.main()
EOF

cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name='my_"$PROJECT_NAME"_project',
    version='0.1',
    packages=find_packages(where="$PROJECT_NAME"),
)
EOF
