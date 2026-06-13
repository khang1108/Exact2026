import os
import shutil
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

WATCHED_PATHS = [
    "requirements.txt",
    "pyproject.toml",
    "src",
    "configs",
    "scripts"
]

KAGGLE_DIR = "kaggle"
ZIP_FILE_NAME = "kaggle"

def get_file_stats(paths):
    stats = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.is_file():
            stats[str(p)] = p.stat().st_mtime
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    file_path = Path(root) / f
                    if "__pycache__" not in str(file_path):
                        stats[str(file_path)] = file_path.stat().st_mtime
    return stats

def sync_and_zip():
    logging.info("Changes detected. Syncing to kaggle folder...")
    
    kaggle_path = Path(KAGGLE_DIR)
    if kaggle_path.exists():
        for item in kaggle_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        kaggle_path.mkdir(parents=True, exist_ok=True)
        
    for path in WATCHED_PATHS:
        p = Path(path)
        if not p.exists():
            logging.warning(f"Warning: {path} does not exist.")
            continue
            
        dest = kaggle_path / p.name
        if p.is_file():
            shutil.copy2(p, dest)
        elif p.is_dir():
            shutil.copytree(p, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            
    logging.info(f"Zipping {KAGGLE_DIR} to {ZIP_FILE_NAME}.zip...")
    shutil.make_archive(ZIP_FILE_NAME, 'zip', KAGGLE_DIR)
    logging.info("Done! Waiting for changes...")

def main():
    logging.info("Starting watcher...")
    last_stats = get_file_stats(WATCHED_PATHS)
    
    sync_and_zip()
    
    try:
        while True:
            time.sleep(2)
            current_stats = get_file_stats(WATCHED_PATHS)
            if current_stats != last_stats:
                sync_and_zip()
                last_stats = current_stats
    except KeyboardInterrupt:
        logging.info("Stopped watching.")

if __name__ == "__main__":
    main()
