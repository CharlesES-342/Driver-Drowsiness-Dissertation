import pandas as pd
from pathlib import Path
 
csv_path = Path(__file__).resolve().parent / 'annotations.csv'
df = pd.read_csv(csv_path)
 
inconsistent_count = 0
absent_count = 0
count_out = 0
for idx, row in df.iterrows():
    implied = 'tired' if row['tired_confidence'] > row['alert_confidence'] else 'alert'
    if str(row['model_prediction']).strip().lower() != implied:
        inconsistent_count += 1
         
print(f"Found {inconsistent_count} inconsistent row(s) out of {len(df)} total rows.")
