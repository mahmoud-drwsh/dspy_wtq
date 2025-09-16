#!/usr/bin/env python3
"""
Simple script to load WTQ test split from the original GitHub repository and log questions and answers
to a JSON log file, one question-answer pair per line.
Based on the original Hugging Face dataset script implementation.
"""

import json
import logging
import csv
import requests
import tempfile
import os
from pathlib import Path

def setup_logging(log_file_path):
    """Set up logging to write JSON formatted entries to file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Only log the message, no timestamp/level
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

def download_wtq_data():
    """Download WTQ data from the original GitHub repository if local file is missing."""
    print("Downloading WTQ data from GitHub repository...")
    
    # URL from the original dataset script
    data_url = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
    
    # Create a temporary directory to store the downloaded file
    temp_dir = tempfile.mkdtemp()
    zip_file_path = os.path.join(temp_dir, "WikiTableQuestions-1.0.2-compact.zip")
    
    try:
        # Download the data
        print(f"Downloading {data_url}...")
        response = requests.get(data_url)
        response.raise_for_status()
        
        # Save the zip file
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded data to {zip_file_path}")
        return zip_file_path
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def extract_wtq_data():
    """Extract WTQ data from local zip file to .cache directory, with fallback to download."""
    print("Extracting WTQ data from local zip file...")
    
    # Path to the local zip file
    script_dir = Path(__file__).parent
    zip_file_path = script_dir / "WikiTableQuestions-1.0.2-compact.zip"
    
    # Create .cache directory in project root
    project_root = script_dir.parent
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Check if local zip file exists, if not download it
    if not zip_file_path.exists():
        print(f"Local zip file not found at {zip_file_path}")
        print("Downloading from GitHub repository as fallback...")
        zip_file_path = download_wtq_data()
    
    # Extract the zip file to .cache
    import zipfile
    try:
        print(f"Extracting {zip_file_path} to {cache_dir}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # The extracted directory should be "WikiTableQuestions"
        extracted_dir = cache_dir / "WikiTableQuestions"
        data_dir = extracted_dir / "data"
        
        print(f"Extracted data to {data_dir}")
        return str(data_dir)
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        raise
    finally:
        # Clean up temporary downloaded file if it was downloaded
        if str(zip_file_path).startswith(tempfile.gettempdir()):
            try:
                os.remove(zip_file_path)
                os.rmdir(os.path.dirname(zip_file_path))
            except:
                pass

def read_table_from_file(table_name, root_dir):
    """Read table content from file, based on the original dataset script."""
    def extract_table_content(line):
        vals = [val.replace("\n", " ").strip() for val in line.strip("\n").split("\t")]
        return vals
    
    rows = []
    # Use the normalized table file (replace .csv with .tsv)
    table_name = table_name.replace(".csv", ".tsv")
    
    # The table files are in subdirectories, so we need to find the correct path
    # Look for the table file in the csv subdirectory
    csv_dir = os.path.join(root_dir, "csv")
    table_path = os.path.join(csv_dir, table_name)
    
    # If not found, try to find it in subdirectories
    if not os.path.exists(table_path):
        for root, dirs, files in os.walk(csv_dir):
            if table_name in files:
                table_path = os.path.join(root, table_name)
                break
    
    with open(table_path, "r", encoding="utf8") as table_f:
        table_lines = table_f.readlines()
        # The first line is header
        header = extract_table_content(table_lines[0])
        for line in table_lines[1:]:
            rows.append(extract_table_content(line))
    
    return {"header": header, "rows": rows, "name": table_name}

def load_wtq_test_split():
    """Load the WTQ test split from the extracted data."""
    print("Loading WTQ test split from extracted data...")
    
    # Extract the data
    data_dir = extract_wtq_data()
    
    # The test file is "pristine-unseen-tables.tsv"
    test_file_path = os.path.join(data_dir, "pristine-unseen-tables.tsv")
    
    test_data = []
    
    try:
        # Read the TSV file, based on the original dataset script
        with open(test_file_path, encoding="utf-8") as f:
            # Skip the first line since it is the tsv header
            next(f)
            for idx, line in enumerate(f):
                example_id, question, table_name, answer = line.strip("\n").split("\t")
                answers = answer.split("|")
                
                # For now, just store the basic info without table content
                # to avoid file path issues
                test_data.append({
                    "id": example_id,
                    "question": question,
                    "answers": answers,
                    "table_name": table_name
                })
        
        print(f"Loaded {len(test_data)} test examples")
        return test_data
        
    except Exception as e:
        print(f"Error reading test data: {e}")
        raise
    finally:
        # No need to clean up since we're using .cache directory
        pass

def log_question_answer(logger, question, answers, example_id=None):
    """Log a single question-answer pair as JSON."""
    # Create a dictionary with the question and answers
    entry = {
        "question": question,
        "answers": answers
    }
    
    # Add example ID if provided
    if example_id is not None:
        entry["id"] = example_id
    
    # Log as JSON string
    logger.info(json.dumps(entry, ensure_ascii=False))

def main():
    """Main function to load WTQ test data and log to file."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    log_file_path = project_root / "test" / "wtq_test_qa.log"
    
    # Ensure test directory exists
    log_file_path.parent.mkdir(exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_file_path)
    
    # Load the test split
    test_data = load_wtq_test_split()
    
    # Log header information
    logger.info(json.dumps({
        "dataset": "WikiTableQuestions",
        "source": "https://github.com/ppasupat/WikiTableQuestions",
        "split": "pristine-unseen-tables",
        "total_examples": len(test_data),
        "description": "WTQ test split questions and answers"
    }))
    
    # Log each question-answer pair
    print(f"Logging {len(test_data)} question-answer pairs to {log_file_path}")
    
    for i, example in enumerate(test_data):
        question = example["question"]
        answers = example["answers"]
        
        # Log the question-answer pair
        log_question_answer(logger, question, answers, example_id=i)
        
        # Print progress every 100 examples
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_data)} examples...")
    
    print(f"Completed! Logged {len(test_data)} question-answer pairs to {log_file_path}")

if __name__ == "__main__":
    main()
