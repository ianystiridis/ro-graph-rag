from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("src/graphrag/output")

for name in [
    "documents",
    "text_units",
    "entities",
    "relationships",
    "communities",
    "community_reports",
]:
    path = OUTPUT_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"\n=== {name} ===")
        print(f"rows: {len(df)}")
        print(df.head(3).to_string())
    else:
        print(f"\n{name}.parquet not found")