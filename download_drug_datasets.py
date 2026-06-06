"""
Download drug datasets via kagglehub (run once to cache locally).
- shaiksha19/drug-dataset
- prathamtripathi/drug-classification
Requires: pip install kagglehub
"""
import sys
from pathlib import Path

# Ensure project root on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def main():
    try:
        import kagglehub
    except ImportError:
        print("Install kagglehub: pip install kagglehub")
        return 1

    print("Downloading drug datasets via kagglehub...")
    print()

    # Drug dataset (shaiksha19/drug-dataset)
    path1 = kagglehub.dataset_download("shaiksha19/drug-dataset")
    print("Path to dataset files (drug-dataset):", path1)
    print()

    # Drug classification (prathamtripathi/drug-classification)
    path2 = kagglehub.dataset_download("prathamtripathi/drug-classification")
    print("Path to dataset files (drug-classification):", path2)
    print()

    print("Done. Use models.drug_data_sources.get_kaggle_drug_dataset_path() / get_kaggle_drug_classification_path() to get paths in code.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
