import os
import shutil
import subprocess
import re
import stat
import time
from typing import List, Dict
from .settings import settings

def _sanitize_dirname(repo_url: str) -> str:
    """Create safe directory name from repo URL"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_url)

def _remove_readonly(func, path, excinfo):
    """Error handler for Windows readonly files during rmtree"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _force_remove_directory(path: str, retries: int = 3):
    """
    Forcefully remove directory with retry logic for Windows.
    Handles readonly files, Git locks, and other Windows-specific issues.
    """
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                # On Windows, use onerror callback to handle readonly files
                shutil.rmtree(path, onerror=_remove_readonly)
                print(f"   Removed cached repository at {path}")
            return True
        except Exception as e:
            if attempt < retries - 1:
                print(f"   Retry {attempt + 1}/{retries - 1}: Cleaning repository cache...")
                time.sleep(0.5)  # Wait before retry
            else:
                print(f"   Warning: Could not fully remove cache: {e}")
                # Try alternative: rename and delete later
                try:
                    temp_path = f"{path}_old_{int(time.time())}"
                    os.rename(path, temp_path)
                    print(f"   Renamed old cache to {temp_path} (manual cleanup needed)")
                except:
                    pass
                return False
    return False

def clone_repository(repo_url: str) -> str:
    """
    Clone GitHub repository and return local path.
    Always fetches fresh copy by removing existing cached repositories.
    """
    os.makedirs(settings.workdir, exist_ok=True)
    local_path = os.path.join(settings.workdir, _sanitize_dirname(repo_url))
    
    # Always remove existing repository to get fresh content
    if os.path.exists(local_path):
        print(f"🧹 Cleaning cached repository...")
        _force_remove_directory(local_path)
    
    try:
        # Clone with depth 1 for faster download
        print(f"⬇️  Cloning fresh repository...")
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

def cleanup_old_repositories(max_age_hours: int = 24):
    """
    Optional: Clean up repositories older than max_age_hours.
    Helps keep _repos folder manageable.
    """
    try:
        if not os.path.exists(settings.workdir):
            return
        
        current_time = time.time()
        removed_count = 0
        
        for dir_name in os.listdir(settings.workdir):
            dir_path = os.path.join(settings.workdir, dir_name)
            if os.path.isdir(dir_path):
                # Check directory age
                dir_age_hours = (current_time - os.path.getmtime(dir_path)) / 3600
                if dir_age_hours > max_age_hours:
                    _force_remove_directory(dir_path)
                    removed_count += 1
        
        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} old cached repositories")
    except Exception as e:
        print(f"Warning: Could not clean old repositories: {e}")
