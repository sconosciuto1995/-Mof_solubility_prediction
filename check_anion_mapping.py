"""
Determine anion ID mapping by checking which anions are in the folder.
The Excel uses numeric IDs 14-17, but files are named by formula.
"""

import os

# List available anion files
anions_dir = 'anions'
anion_files = sorted([f.replace('.csv', '') for f in os.listdir(anions_dir) if f.endswith('.csv')])
print("Available anions (by file name):")
for i, anion in enumerate(anion_files, 1):
    print(f"  {i}: {anion}")

print(f"\nTotal: {len(anion_files)} anion types")

# The Excel data shows anion rows 14-17, but we only have 12 anion files
# This suggests either:
# 1. The Excel is using a different numbering scheme
# 2. There's a mapping file we're missing
# 3. The anion IDs in the Excel correspond to row indices, not chemical formulas

print("\n" + "="*80)
print("QUESTION: What is the mapping from numeric IDs (14-17 in Excel) to anion names?")
print("="*80)
print("""
Possibilities:
1. IDs 14-17 might be experiment numbers, not anion IDs
2. There might be a separate mapping file
3. We need to ask which anion corresponds to which ID

The 12 available anions are:
""")
for anion in anion_files:
    print(f"  - {anion}")
