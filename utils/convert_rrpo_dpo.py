#!/usr/bin/env python3
"""
Script to convert RRPO dataset to DPO dataset format by removing <mask> and </mask> tags
from the chosen and rejected fields.
"""

import json
import re
import argparse
from pathlib import Path


def remove_mask_tags(text):
    """
    Remove <mask> and </mask> tags from text and clean up extra spaces.
    
    Args:
        text (str): Input text containing mask tags
        
    Returns:
        str: Text with mask tags removed and extra spaces collapsed
    """
    if not isinstance(text, str):
        return text
    
    # Remove <mask> and </mask> tags, keeping the content inside
    text = re.sub(r'</?mask>', '', text)
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Clean up leading/trailing spaces
    text = text.strip()
    
    return text


def convert_rrpo_to_dpo(rrpo_file, dpo_file):
    """
    Convert RRPO dataset to DPO format by removing mask tags.
    
    Args:
        rrpo_file (str): Path to input RRPO dataset JSON file
        dpo_file (str): Path to output DPO dataset JSON file
    """
    print(f"Reading RRPO dataset from: {rrpo_file}")
    
    # Read the RRPO dataset
    with open(rrpo_file, 'r', encoding='utf-8') as f:
        rrpo_data = json.load(f)
    
    print(f"Loaded {len(rrpo_data)} entries from RRPO dataset")
    
    # Convert each entry
    dpo_data = []
    id_qid_added = set()
    for i, entry in enumerate(rrpo_data):
        if i % 1000 == 0:
            print(f"Processing entry {i+1}/{len(rrpo_data)}")
        
        if (entry['id'], entry['qid']) in id_qid_added:
            continue
        id_qid_added.add((entry['id'], entry['qid']))
        
        # if not entry.get('was_correct', False):
        if True:
            dpo_entry = {
                'id': entry['id'],
                'qid': entry['qid'],
                'prompt': entry['prompt'],
                'chosen': remove_mask_tags(entry['chosen']),
                'rejected': remove_mask_tags(entry['rejected']),
                'image_path': entry['image_path'],
                'was_correct': entry.get('was_correct', False)
            }
            dpo_data.append(dpo_entry)
    
    # Write the DPO dataset
    print(f"Writing DPO dataset to: {dpo_file}")
    with open(dpo_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed! {len(dpo_data)} entries written to DPO dataset")


def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(
        description="Convert RRPO dataset to DPO format by removing mask tags"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input RRPO dataset JSON file (default: data/rrpo_dataset.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='',
        help='Path to output DPO dataset JSON file (default: data/new_dpo_dataset.json)'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate that the conversion was successful by checking a few entries'
    )
    
    args = parser.parse_args()
    # args.output = args.input.replace('rrpo', 'dpo')
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform conversion
    try:
        convert_rrpo_to_dpo(args.input, args.output)
        
        # Optional validation
        if args.validate:
            print("\nValidating conversion...")
            validate_conversion(args.input, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


def validate_conversion(rrpo_file, dpo_file):
    """
    Validate that the conversion was successful by checking a few entries.
    
    Args:
        rrpo_file (str): Path to original RRPO dataset
        dpo_file (str): Path to converted DPO dataset
    """
    with open(rrpo_file, 'r', encoding='utf-8') as f:
        rrpo_data = json.load(f)
    
    with open(dpo_file, 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    print(f"Original RRPO entries: {len(rrpo_data)}")
    print(f"Converted DPO entries: {len(dpo_data)}")
    
    if len(rrpo_data) != len(dpo_data):
        print("ERROR: Mismatch in number of entries!")
        return
    
    # Check a few random entries
    import random
    sample_indices = random.sample(range(len(rrpo_data)), min(5, len(rrpo_data)))
    
    for idx in sample_indices:
        rrpo_entry = rrpo_data[idx]
        dpo_entry = dpo_data[idx]
        
        print(f"\nChecking entry {idx} (ID: {rrpo_entry['id']}):")
        
        # Check that mask tags are removed
        original_chosen = rrpo_entry['chosen']
        converted_chosen = dpo_entry['chosen']
        
        if '<mask>' in converted_chosen or '</mask>' in converted_chosen:
            print(f"ERROR: Mask tags still present in chosen field!")
            print(f"Original: {original_chosen[:100]}...")
            print(f"Converted: {converted_chosen[:100]}...")
        else:
            print("✓ Mask tags successfully removed from chosen field")
            print(f"  Before: {original_chosen[:80]}...")
            print(f"  After:  {converted_chosen[:80]}...")
        
        original_rejected = rrpo_entry['rejected']
        converted_rejected = dpo_entry['rejected']
        
        if '<mask>' in converted_rejected or '</mask>' in converted_rejected:
            print(f"ERROR: Mask tags still present in rejected field!")
            print(f"Original: {original_rejected[:100]}...")
            print(f"Converted: {converted_rejected[:100]}...")
        else:
            print("✓ Mask tags successfully removed from rejected field")
            print(f"  Before: {original_rejected[:80]}...")
            print(f"  After:  {converted_rejected[:80]}...")
        
        # Check that other fields are preserved
        if (rrpo_entry['prompt'] == dpo_entry['prompt'] and 
            rrpo_entry['image_path'] == dpo_entry['image_path']):
            print("✓ Other fields preserved correctly")
        else:
            print("ERROR: Other fields not preserved correctly!")


if __name__ == "__main__":
    exit(main())
