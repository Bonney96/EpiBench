import os
from datetime import datetime

# Root directory containing the code
ROOT_DIR = "/storage1/fs1/dspencer/Active/spencerlab/abonney/EpiBench"

# File extensions to include in the snapshot
FILE_EXTENSIONS = ['.py', '.sh', '.R', '.ipynb', '.yml', '.yaml']

def should_include(file_name):
    # Exclude macOS resource fork files that start with '._'
    if file_name.startswith("._"):
        return False
    # Include file if it has one of the allowed extensions
    return any(file_name.lower().endswith(ext) for ext in FILE_EXTENSIONS)

def create_text_snapshot(root_dir, extensions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"code_snapshot_{timestamp}.txt"
    snapshot_path = os.path.join(os.getcwd(), snapshot_name)

    with open(snapshot_path, 'w', encoding='utf-8', errors='replace') as out_file:
        # Recursively walk the root directory
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if should_include(file):
                    full_path = os.path.join(root, file)
                    # Write a header for each file before its content
                    relative_path = os.path.relpath(full_path, start=root_dir)
                    out_file.write("\n" + "="*80 + "\n")
                    out_file.write(f"FILE: {relative_path}\n")
                    out_file.write("="*80 + "\n")

                    # Append file contents
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        out_file.write(content)
                        out_file.write("\n")  # Blank line after file content
                    except Exception as e:
                        out_file.write(f"\n[Error reading file: {e}]\n")

    print(f"Text snapshot created: {snapshot_path}")

if __name__ == "__main__":
    create_text_snapshot(ROOT_DIR, FILE_EXTENSIONS)
