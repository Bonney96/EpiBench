import os
import tarfile
from datetime import datetime

# Root directory containing the code
ROOT_DIR = "/storage1/fs1/dspencer/Active/spencerlab/abonney/EpiBench"

# File extensions to include in the snapshot
FILE_EXTENSIONS = ['.py', '.sh', '.R', '.ipynb', '.yml', '.yaml', '.json', '.txt', '.md']

def should_include(file_name):
    # Include file if it has one of the allowed extensions
    return any(file_name.lower().endswith(ext) for ext in FILE_EXTENSIONS)

def create_snapshot(root_dir, extensions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"code_snapshot_{timestamp}.tar.gz"
    snapshot_path = os.path.join(os.getcwd(), snapshot_name)

    # Open the tar.gz file for writing
    with tarfile.open(snapshot_path, "w:gz") as tar:
        # Recursively walk the root directory
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if should_include(file):
                    full_path = os.path.join(root, file)
                    # Add file to the archive, preserving directory structure relative to ROOT_DIR
                    arcname = os.path.relpath(full_path, start=root_dir)
                    tar.add(full_path, arcname=arcname)

    print(f"Snapshot created: {snapshot_path}")

if __name__ == "__main__":
    create_snapshot(ROOT_DIR, FILE_EXTENSIONS)
