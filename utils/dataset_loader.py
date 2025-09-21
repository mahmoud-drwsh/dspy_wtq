"""
Dataset loading utilities for WikiTableQuestions (WTQ).

Handles downloading and extracting the WTQ dataset from the official repository.
Provides functions to ensure the dataset is available locally without manual intervention.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

import requests


def download_wtq_zip() -> Path:
    """Download the WTQ compact zip into ./setup/ and return its path.

    If the file already exists in ./setup/, no download occurs.
    
    Returns:
        Path: Path to the downloaded zip file
    """
    data_url = (
        "https://github.com/ppasupat/WikiTableQuestions/releases/download/"
        "v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    )

    project_root = Path(__file__).resolve().parent.parent  # utils/ -> project root
    setup_dir = project_root / "setup"
    setup_dir.mkdir(exist_ok=True)
    zip_file_path = setup_dir / "WikiTableQuestions-1.0.2-compact.zip"

    if zip_file_path.exists():
        return zip_file_path

    # Stream to file to avoid holding the entire zip in memory
    with requests.get(data_url, stream=True) as resp:
        resp.raise_for_status()
        with open(zip_file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return zip_file_path


def ensure_wtq_data(zip_file_path: Optional[Path] = None, cache_dir: Optional[Path] = None) -> Path:
    """Ensure WTQ data is extracted locally and return the `data/` directory path.

    Args:
        zip_file_path: Path to the zip file. If None, looks for it in ./setup/
        cache_dir: Directory to extract to. If None, uses project .cache/
        
    Returns:
        Path: Path to the extracted WikiTableQuestions/data directory
    """
    # Determine a sensible default root based on this file location
    script_dir = Path(__file__).resolve().parent.parent  # utils/ -> project root
    project_root = script_dir

    if zip_file_path is None:
        # Assume a copy may live next to the setup script (project_root/setup/)
        candidate = project_root / "setup" / "WikiTableQuestions-1.0.2-compact.zip"
        zip_file_path = candidate

    cache_dir = cache_dir or (project_root / ".cache")
    cache_dir.mkdir(exist_ok=True)

    existing_data_dir = cache_dir / "WikiTableQuestions" / "data"
    if existing_data_dir.exists():
        return existing_data_dir

    # Ensure a zip exists (download if missing)
    if not Path(zip_file_path).exists():
        zip_file_path = download_wtq_zip()

    # Extract the zip file to cache_dir
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(cache_dir)

    data_dir = cache_dir / "WikiTableQuestions" / "data"
    return data_dir


def get_wtq_root_dir(data_dir: Optional[Path] = None) -> Path:
    """Get the root directory of the WTQ dataset (parent of data/ directory).
    
    Args:
        data_dir: Path to the data directory. If None, ensures data is available.
        
    Returns:
        Path: Root directory containing csv/, data/, page/, etc.
    """
    if data_dir is None:
        data_dir = ensure_wtq_data()
    
    # The data directory is WikiTableQuestions/data, so parent is the root
    return data_dir.parent


def is_wtq_data_available() -> bool:
    """Check if WTQ data is already extracted and available.
    
    Returns:
        bool: True if data is available, False otherwise
    """
    script_dir = Path(__file__).resolve().parent.parent
    cache_dir = script_dir / ".cache"
    data_dir = cache_dir / "WikiTableQuestions" / "data"
    return data_dir.exists()


def get_wtq_data_path() -> Optional[Path]:
    """Get the path to the WTQ data directory if it exists.
    
    Returns:
        Path: Path to data directory if available, None otherwise
    """
    script_dir = Path(__file__).resolve().parent.parent
    cache_dir = script_dir / ".cache"
    data_dir = cache_dir / "WikiTableQuestions" / "data"
    return data_dir if data_dir.exists() else None
