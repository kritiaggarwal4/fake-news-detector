"""
download_datasets.py
────────────────────
Run this ONCE before starting the app to download datasets.

Usage:
    python download_datasets.py

What it downloads
─────────────────
1. LIAR dataset  →  data/liar/   (train.tsv, valid.tsv, test.tsv)
2. Kaggle Fake-and-Real News → data/kaggle/ (needs manual download — see instructions)
3. FakeNewsNet  → data/fakenewsnet/ (needs manual download — see instructions)
"""

import os
import zipfile
import urllib.request

# ── LIAR Dataset (auto-download) ──────────────────────────────────────────────
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
LIAR_DIR = os.path.join("data", "liar")

def download_liar():
    os.makedirs(LIAR_DIR, exist_ok=True)
    already = all(
        os.path.exists(os.path.join(LIAR_DIR, f))
        for f in ["train.tsv", "valid.tsv", "test.tsv"]
    )
    if already:
        print("✅ LIAR dataset already present.")
        return

    print("⬇️  Downloading LIAR dataset (~3 MB)…")
    try:
        zip_path = os.path.join("data", "liar.zip")
        urllib.request.urlretrieve(LIAR_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(LIAR_DIR)
        os.remove(zip_path)
        print(f"✅ LIAR dataset saved to  {LIAR_DIR}/")
    except Exception as e:
        print(f"❌ Failed to download LIAR: {e}")
        print("   Manual download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
        print(f"   Extract train.tsv / valid.tsv / test.tsv into  {LIAR_DIR}/")

# ── Kaggle Dataset (manual) ───────────────────────────────────────────────────
def kaggle_instructions():
    kaggle_dir = os.path.join("data", "kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    if os.path.exists(os.path.join(kaggle_dir, "Fake.csv")):
        print("✅ Kaggle dataset already present.")
        return

    print("""
📋  KAGGLE DATASET — Manual download required
─────────────────────────────────────────────
1. Go to:  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click  Download  (you need a free Kaggle account)
3. Unzip the downloaded archive
4. Copy  Fake.csv  and  True.csv  into:
         {}/
5. Re-run this script to confirm.

(This dataset has ~44,000 articles and significantly boosts accuracy.)
""".format(kaggle_dir))

# ── FakeNewsNet (manual) ──────────────────────────────────────────────────────
def fakenewsnet_instructions():
    fnn_dir = os.path.join("data", "fakenewsnet")
    os.makedirs(fnn_dir, exist_ok=True)
    already = any(
        os.path.exists(os.path.join(fnn_dir, f))
        for f in ["politifact_fake.csv", "politifact_real.csv"]
    )
    if already:
        print("✅ FakeNewsNet dataset already present.")
        return

    print("""
📋  FAKENEWSNET DATASET — Manual download required
──────────────────────────────────────────────────
1. Go to:  https://github.com/KaiDMML/FakeNewsNet
2. Download the CSV files from the repository or Google Drive link in README
   Files needed:
     • politifact_fake.csv
     • politifact_real.csv
     • gossipcop_fake.csv   (optional but recommended)
     • gossipcop_real.csv   (optional but recommended)
3. Copy them into:
         {}/
4. Re-run this script to confirm.
""".format(fnn_dir))

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_liar()
    kaggle_instructions()
    fakenewsnet_instructions()

    print("\n🎉 Done! Now run:  streamlit run app.py")
    print("   On first launch the model trains automatically and caches to model_cache.pkl")