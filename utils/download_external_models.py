"""
Download External Models from GitHub and Google Drive
Handles models not available on Hugging Face
"""

import os
import requests
import zipfile
import json
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


# External model sources from DATASET_MODEL_LINKS.md
EXTERNAL_MODELS = {
    "medmcqa": {
        "source": "google_drive",
        "url": "https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky",
        "type": "dataset",
        "destination": "medmcqa_data/",
        "description": "194K Multiple Choice Questions"
    },
    # Add more external models as needed
}


def download_file(url: str, destination: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Path to save file
        chunk_size: Chunk size for download
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_google_drive_file(file_id: str, destination: Path):
    """
    Download file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        destination: Path to save file
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # First request to get download link
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Handle large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            url += f"&confirm={value}"
            break
    
    download_file(url, destination)


def download_github_repo(repo_url: str, destination: Path, branch: str = "main"):
    """
    Download a GitHub repository
    
    Args:
        repo_url: GitHub repository URL
        destination: Path to save repository
        branch: Branch to download
    """
    # Convert GitHub URL to download URL
    if "github.com" in repo_url:
        repo_path = repo_url.replace("https://github.com/", "").replace(".git", "")
        download_url = f"https://github.com/{repo_path}/archive/refs/heads/{branch}.zip"
        
        zip_path = destination / f"{repo_path.split('/')[-1]}.zip"
        download_file(download_url, zip_path)
        
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        
        # Remove zip
        zip_path.unlink()
        
        logger.info(f"✅ Downloaded and extracted {repo_url}")
    else:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")


def download_external_model(model_key: str, base_dir: Optional[Path] = None) -> Path:
    """
    Download an external model
    
    Args:
        model_key: Key from EXTERNAL_MODELS
        base_dir: Base directory for downloads
        
    Returns:
        Path to downloaded model
    """
    if model_key not in EXTERNAL_MODELS:
        raise ValueError(f"Unknown external model: {model_key}")
    
    model_info = EXTERNAL_MODELS[model_key]
    base_dir = base_dir or Path(__file__).parent.parent / "external_models"
    destination = base_dir / model_info["destination"]
    
    if destination.exists():
        logger.info(f"✅ {model_key} already exists at {destination}")
        return destination
    
    logger.info(f"📥 Downloading {model_key}...")
    
    if model_info["source"] == "google_drive":
        # Extract file ID from URL
        file_id = model_info["url"].split("id=")[-1]
        download_google_drive_file(file_id, destination)
    elif model_info["source"] == "github":
        download_github_repo(model_info["url"], destination)
    else:
        download_file(model_info["url"], destination)
    
    logger.info(f"✅ Downloaded {model_key} to {destination}")
    return destination


def download_all_external_models(base_dir: Optional[Path] = None):
    """Download all external models"""
    base_dir = base_dir or Path(__file__).parent.parent / "external_models"
    
    logger.info("="*70)
    logger.info("DOWNLOADING EXTERNAL MODELS")
    logger.info("="*70)
    
    results = {"success": [], "failed": []}
    
    for model_key in EXTERNAL_MODELS.keys():
        try:
            path = download_external_model(model_key, base_dir)
            results["success"].append((model_key, path))
        except Exception as e:
            logger.error(f"❌ Failed to download {model_key}: {str(e)}")
            results["failed"].append((model_key, str(e)))
    
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*70)
    logger.info(f"\n✅ Successfully downloaded: {len(results['success'])}")
    for model_key, path in results["success"]:
        logger.info(f"   - {model_key}: {path}")
    
    if results["failed"]:
        logger.info(f"\n❌ Failed: {len(results['failed'])}")
        for model_key, error in results["failed"]:
            logger.info(f"   - {model_key}: {error}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download external models")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all external models"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.model:
        download_external_model(args.model)
    elif args.all:
        download_all_external_models()
    else:
        print("Available external models:")
        for key, info in EXTERNAL_MODELS.items():
            print(f"  - {key}: {info['description']}")
        print("\nUse --model <key> to download a specific model")
        print("Use --all to download all external models")

