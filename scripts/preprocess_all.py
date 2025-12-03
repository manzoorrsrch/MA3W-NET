import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.data.preprocess import process_case

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True,
                    help="CSV containing case_id and path entries")
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocess cases"):
        process_case(row, args.cache_dir)

if __name__ == "__main__":
    main()
