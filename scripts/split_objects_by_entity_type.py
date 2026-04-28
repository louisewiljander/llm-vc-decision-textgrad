#!/usr/bin/env python
"""
Split objects.csv into separate files by entity_type.

Creates:
  - data/raw/companies.csv (entity_type='Company')
  - data/raw/financial_orgs.csv (entity_type='FinancialOrg')
  - data/raw/products.csv (entity_type='Product')

Note: Person entities are not extracted; use existing data/raw/people.csv instead.
"""

import pandas as pd
from pathlib import Path

# Define paths
DATA_RAW = Path("data/raw")
OBJECTS_PATH = DATA_RAW / "objects.csv"

# Map entity_type → output filename
ENTITY_TYPE_MAP = {
    "Company": "companies.csv",
    "FinancialOrg": "financial_orgs.csv",
    "Product": "products.csv",
}

def split_objects():
    """Load objects.csv and split by entity_type."""
    print(f"Loading {OBJECTS_PATH}...")
    objects = pd.read_csv(OBJECTS_PATH, low_memory=False)
    
    print(f"Total rows: {len(objects):,}")
    print("\nEntity type distribution:")
    print(objects['entity_type'].value_counts().to_string())
    
    for entity_type, filename in ENTITY_TYPE_MAP.items():
        subset = objects[objects['entity_type'] == entity_type].copy()
        output_path = DATA_RAW / filename
        subset.to_csv(output_path, index=False)
        print(f"\n✓ {filename}: {len(subset):,} rows")
        print(f"  → {output_path}")
    
    print("\nDone! Entity-type splits created in data/raw/")

if __name__ == "__main__":
    split_objects()
