import pandas as pd
import re
from collections import defaultdict
import sys

if len(sys.argv) > 1:
    parameter = sys.argv[1]

def extract_all_fields(csv_file):
    """
    Extract all unique field names from the entire audit log file
    """
    print("Scanning file to discover all possible fields...")
    all_fields = set()
    
    with open(csv_file, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                # Split on first comma to separate label from audit log content
                if ',' in line:
                    label = line.split(',', 1)[0].strip()
                    content = line.split(',', 1)[1].strip()
                    
                    # Use regex to find all field=value patterns
                    # This pattern captures field names before = signs
                    field_matches = re.findall(r'(\w+)=', content)
                    
                    for field in field_matches:
                        all_fields.add(field)
    
    print(f"Discovered {len(all_fields)} unique fields")
    return sorted(all_fields)

def parse_single_line(content, all_fields):
    """
    Parse a single audit log line and extract all field values
    The key insight: we need to find field=value pairs where value continues
    until the next field= pattern or end of string
    """
    features = {field: '' for field in all_fields}
    
    # Create a pattern that matches field=value where value continues until next field= or end
    # This handles complex values like: exit=ENOENT(No,such,file,or,directory)
    pattern = r'(\w+)=([^,]*(?:,[^,]*)*?)(?=\s+\w+=|\s*$)'
    
    # Alternative approach: split the content into field=value segments
    # Find all positions where field= patterns start
    field_positions = []
    for match in re.finditer(r'\b(\w+)=', content):
        field_positions.append((match.start(), match.group(1)))
    
    # Extract values between field positions
    for i, (start_pos, field_name) in enumerate(field_positions):
        if field_name in all_fields:
            # Find where this field's value ends (start of next field or end of string)
            if i + 1 < len(field_positions):
                end_pos = field_positions[i + 1][0]
                field_content = content[start_pos:end_pos].strip()
            else:
                field_content = content[start_pos:].strip()
            
            # Extract the value part (everything after field=)
            if '=' in field_content:
                value = field_content.split('=', 1)[1].strip()
                # Remove trailing comma if it exists (but preserve commas within values)
                if value.endswith(',') and not value.endswith(',,'):
                    value = value[:-1]
                features[field_name] = value
    
    return features

def robust_standardize_dataset(input_csv, output_csv):
    """
    Robustly standardize the audit log dataset
    """
    print("=== ROBUST AUDIT LOG STANDARDIZATION ===")
    
    # Step 1: Discover all fields
    all_fields = extract_all_fields(input_csv)
    print(f"\nDiscovered fields: {all_fields[:10]}{'...' if len(all_fields) > 10 else ''}")
    
    # Step 2: Process each line
    standardized_data = []
    
    with open(input_csv, "r", encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    # Split label from content
                    if ',' in line:
                        label = line.split(',', 1)[0].strip()
                        content = line.split(',', 1)[1].strip()
                        
                        # Parse all fields from this line
                        features = parse_single_line(content, all_fields)
                        features['label'] = label
                        
                        standardized_data.append(features)
                        
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} lines...")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    print(f"Line content: {line.strip()[:100]}...")
                    continue
    
    print(f"Successfully processed {len(standardized_data)} lines")
    
    # Step 3: Create DataFrame
    df = pd.DataFrame(standardized_data)
    
    # Ensure label is first column, then alphabetical order for others
    columns = ['label'] + sorted([col for col in df.columns if col != 'label'])
    df = df[columns]
    
    # Step 4: Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\nDataset standardized successfully!")
    print(f"Shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Output saved to: {output_csv}")
    
    return df

def analyze_parsing_quality(df, sample_size=5):
    """
    Analyze the quality of the parsing
    """
    print(f"\n=== PARSING QUALITY ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Show columns with most data
    print(f"\nTop 15 columns with most non-empty values:")
    non_empty_counts = (df != '').sum().sort_values(ascending=False)
    print(non_empty_counts.head(15))
    
    # Show sample parsed data
    print(f"\n=== SAMPLE PARSED ENTRIES ===")
    for i in range(min(sample_size, len(df))):
        print(f"\nEntry {i+1} - Label: {df.iloc[i]['label']}")
        # Show non-empty fields for this entry
        row_data = df.iloc[i]
        non_empty_fields = {k: v for k, v in row_data.items() 
                           if v != '' and k != 'label'}
        
        # Show first 8 non-empty fields to avoid overwhelming output
        shown_count = 0
        for field, value in non_empty_fields.items():
            if shown_count < 8:
                print(f"  {field}: {value}")
                shown_count += 1
            else:
                break
        
        if len(non_empty_fields) > 8:
            print(f"  ... and {len(non_empty_fields) - 8} more fields")
    
    return df

def test_specific_cases(df):
    """
    Test specific challenging cases to verify parsing quality
    """
    print(f"\n=== TESTING SPECIFIC CASES ===")
    
    # Test exit field parsing (should handle complex values)
    exit_values = df[df['exit'] != '']['exit'].unique()[:5]
    print(f"Sample 'exit' field values:")
    for val in exit_values:
        print(f"  '{val}'")
    
    # Test proctitle field parsing
    proctitle_values = df[df['proctitle'] != '']['proctitle'].unique()[:3]
    print(f"\nSample 'proctitle' field values:")
    for val in proctitle_values:
        print(f"  '{val}'")

# Main execution
if __name__ == "__main__":
    # Run the standardization
    df = robust_standardize_dataset(f'/home/ubuntu/Auditd_AI/data/{parameter}.csv', '/home/ubuntu/Auditd_AI/data/standardised_data.csv')
    
    # Analyze the results
    analyze_parsing_quality(df)
    
    # Test specific cases
    test_specific_cases(df)