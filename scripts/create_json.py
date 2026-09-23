import pdfplumber
import os
import json
import re

# Project root = parent of this scripts/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define input and output directories
ROOT_DIR = os.path.join(PROJECT_ROOT, "pdfs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output_json")
FOLDERS = ["bad_profile", "good_profile", "excellent_profile"]

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_assets_from_outcome(outcome_text):
    """Extract and combine Technical and Non-Technical Assets from Function Outcome text."""
    assets = []
    lines = outcome_text.replace("\n", " ").strip().split("-")
    tech_assets = []
    non_tech_assets = []
    current_list = None
    
    # Flexible regex patterns for markers
    tech_marker = re.compile(r"(Technical\s*Assets|Tech\s*Assets)", re.IGNORECASE)
    non_tech_marker = re.compile(r"(Non-Technical\s*Assets|Non\s*Tech\s*Assets)", re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if tech_marker.search(line):
            current_list = tech_assets
        elif non_tech_marker.search(line):
            current_list = non_tech_assets
        elif current_list is not None and line:
            cleaned_line = re.sub(r"\s+", " ", line).strip()
            if cleaned_line:
                current_list.append(cleaned_line)
    
    assets = list(set(tech_assets + non_tech_assets))
    return assets

def process_pdf(pdf_path, quality_label):
    """Extract Quality and Function Outcome from a PDF and return structured data."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Try table extraction first
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table[1:]:
                        if len(row) < 4:
                            continue
                        outcome = row[3]
                        if outcome:
                            assets = extract_assets_from_outcome(outcome)
                            if assets:
                                return {
                                    "quality": quality_label.lower(),
                                    "function_outcome": assets
                                }
            
            # Fallback: Extract text from all pages
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "
            
            if full_text:
                # Try to find Technical and Non-Technical Assets
                assets = extract_assets_from_outcome(full_text)
                if assets:
                    return {
                        "quality": quality_label.lower(),
                        "function_outcome": assets
                    }
                else:
                    print(f"Warning: No assets found in text for {pdf_path}")
                    print(f"Sample text: {full_text[:200]}...")  # Log sample for debugging
            else:
                print(f"Warning: No text extracted from {pdf_path}")
        
        print(f"Error: No valid data extracted from {pdf_path}")
        return None
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def main():
    """Process all PDFs in bad_profile, good_profile, excellent_profile folders and save as JSON."""
    for folder in FOLDERS:
        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue
        
        for pdf_file in os.listdir(folder_path):
            if not pdf_file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"Processing {pdf_path}...")
            
            data = process_pdf(pdf_path, quality_label=folder.split("_")[0])  # Use 'bad', 'good', 'excellent'
            if data:
                output_filename = os.path.splitext(pdf_file)[0] + ".json"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Saved JSON to {output_path}")
            else:
                print(f"Skipped {pdf_file} due to extraction failure")

if __name__ == "__main__":
    main()