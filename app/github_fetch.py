import os
import shutil
import subprocess
import re
from typing import List, Dict
from .settings import settings

def _sanitize_dirname(repo_url: str) -> str:
    """Create safe directory name from repo URL"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_url)

def clone_repository(repo_url: str) -> str:
    """Clone GitHub repository and return local path"""
    os.makedirs(settings.workdir, exist_ok=True)
    local_path = os.path.join(settings.workdir, _sanitize_dirname(repo_url))
    
    # Remove if exists
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    try:
        # Clone with depth 1 for faster download
        subprocess.check_call(
            ["git", "clone", "--depth", "1", repo_url, local_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return local_path
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to clone repository: {str(e)}")

def collect_python_files(root_path: str) -> List[str]:
    """Collect all .py files excluding tests and common ignored directories"""
    python_files = []
    ignore_dirs = {"test", "tests", "__pycache__", "venv", "env", ".git", "node_modules"}
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out ignored directories
        dirnames[:] = [d for d in dirnames if d.lower() not in ignore_dirs]
        
        for filename in filenames:
            if filename.endswith(".py") and not filename.startswith("test_"):
                python_files.append(os.path.join(dirpath, filename))
    
    return python_files

def read_source_files(file_paths: List[str]) -> Dict[str, str]:
    """Read source code from files"""
    sources = {}
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                sources[path] = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f:
                    sources[path] = f.read()
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
                continue
    return sources
