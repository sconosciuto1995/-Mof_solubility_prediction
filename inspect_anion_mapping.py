"""
Inspect the anion_mapping.xlsx file to understand the structure.
"""

import pandas as pd
import os

current_dir = os.getcwd()
mapping_path = os.path.join(current_dir, 'anion_mapping.xlsx')

print("="*80)
print("ANION MAPPING FILE STRUCTURE")
print("="*80)

try:
    xl = pd.ExcelFile(mapping_path)
    print(f"Sheet names: {xl.sheet_names}\n")
    
    for sheet_name in xl.sheet_names:
        print(f"--- Sheet: {sheet_name} ---")
        df = pd.read_excel(mapping_path, sheet_name=sheet_name)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nData:\n{df}")
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("="*80)
