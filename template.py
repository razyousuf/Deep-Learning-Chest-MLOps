import os
from pathlib import Path
import logging

# Set up logging format and level
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the project name (used to name the src module)
project_name = "cnnClassifier"

# List of relative paths for directories and files to be created
list_of_files = [
    f"src/{project_name}/__init__.py",                          # main module init
    f"src/{project_name}/components/__init__.py",               # components submodule
    f"src/{project_name}/utils/__init__.py",                    # utils submodule
    f"src/{project_name}/config/__init__.py",                   # config submodule
    f"src/{project_name}/config/configuration.py",              # main configuration logic
    f"src/{project_name}/pipeline/__init__.py",                 # training/inference pipelines
    f"src/{project_name}/entity/__init__.py",                   # data classes / config entities
    f"src/{project_name}/constants/__init__.py",                # constants file
    "config/config.yaml",                                       # configuration YAML
    "dvc.yaml",                                                 # DVC pipeline definition
    "params.yaml",                                              # model/pipeline parameters
    "requirements.txt",                                         # list of dependencies
    "setup.py",                                                 # packaging configuration
    "research/trials.ipynb",                                    # experimental notebook
    "templates/index.html"                                      # sample HTML file (for Flask web app)
]

# Iterate over the defined file paths
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to a Path object for OS-independent handling
    filedir, filename = os.path.split(filepath)  # Split path into directory and file, e.g., "src/cnnClassifier" and "__init__.py"

    # Create the directory if it doesn't already exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create the file only if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Empty file placeholder
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")
