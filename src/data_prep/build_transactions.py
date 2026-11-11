"""
Phase 2 — ساخت تراکنش و کاتالوگ محصول از داده خام Amazon Reviews 2023

ورودی:
  - data/raw/{CATEGORY}.csv.gz              (نسخه 5-core: user_id, parent_asin, rating, timestamp)
  - data/raw/meta_{CATEGORY}.jsonl.gz        (متادیتای محصول، خط به خط JSON)

خروجی:
  - data/processed/{CATEGORY}_transactions.csv   (user_id, items  -> items با '|' جدا شده)
  - data/processed/{CATEGORY}_products.csv       (parent_asin, title, price, brand, store, category)

استفاده:
    python src/data_prep/build_transactions.py --category Cell_Phones_and_Accessories
"""

import argparse
import gzip
import json
import os

import pandas as pd

MIN_RATING = 4          # فقط رتبه‌بندی‌های ۴ و ۵ به‌عنوان "خرید مثبت" در نظر گرفته می‌شن
MAX_USERS = 8000         # پیش‌فرض؛ با --max-users قابل تغییره
MIN_ITEMS_PER_USER = 2   # کاربرانی با کمتر از ۲ آیتم برای ARM بی‌فایده‌ن، حذف می‌شن


def load_ratings(raw_dir: str, category: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, f"{category}.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    print(f"[ratings] loaded {len(df):,} rows from {path}")
    return df


def build_transactions(df: pd.DataFrame, max_users: int = MAX_USERS) -> pd.DataFrame:
    df = df[df["rating"] >= MIN_RATING].copy()

    # پرتعامل‌ترین کاربران رو نگه می‌داریم (نه نمونه تصادفی، چون کاربران کم‌تعامل برای ARM اطلاعات کمی دارن)
    user_counts = df.groupby("user_id").size().sort_values(ascending=False)
    top_users = user_counts.head(max_users).index
    df = df[df["user_id"].isin(top_users)]

    # هر کاربر ممکنه چند بار یک محصول رو ریویو کرده باشه؛ یکتا می‌کنیم
    grouped = (
        df.drop_duplicates(subset=["user_id", "parent_asin"])
        .groupby("user_id")["parent_asin"]
        .apply(list)
        .reset_index(name="items")
    )
    grouped = grouped[grouped["items"].apply(len) >= MIN_ITEMS_PER_USER]

    print(f"[transactions] {len(grouped):,} users kept "
          f"(avg {grouped['items'].apply(len).mean():.1f} items/user)")
    return grouped


def load_products_meta(raw_dir: str, category: str, needed_asins: set) -> pd.DataFrame:
    path = os.path.join(raw_dir, f"meta_{category}.jsonl.gz")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            asin = rec.get("parent_asin")
            if asin not in needed_asins:
                continue  # فقط محصولاتی که واقعاً در تراکنش‌های ما هستن رو نگه می‌داریم (صرفه‌جویی در حافظه)
            details = rec.get("details") or {}
            rows.append({
                "parent_asin": asin,
                "title": rec.get("title"),
                "price": rec.get("price"),
                "brand": details.get("Brand"),
                "store": rec.get("store"),
                "category": " > ".join(rec.get("categories") or []) or rec.get("main_category"),
            })
    products = pd.DataFrame(rows).drop_duplicates(subset=["parent_asin"])
    print(f"[products] matched metadata for {len(products):,} / {len(needed_asins):,} needed items")
    missing = len(needed_asins) - len(products)
    if missing > 0:
        print(f"[products] WARNING: {missing:,} items had no metadata record — "
              f"these will need a placeholder or should be dropped before ARM (Phase 3)")
    return products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                         help="e.g. Cell_Phones_and_Accessories or Electronics")
    parser.add_argument("--max-users", type=int, default=MAX_USERS)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ratings = load_ratings(args.raw_dir, args.category)
    transactions = build_transactions(ratings, args.max_users)

    needed_asins = set(a for items in transactions["items"] for a in items)
    products = load_products_meta(args.raw_dir, args.category, needed_asins)

    tx_path = os.path.join(args.out_dir, f"{args.category}_transactions.csv")
    pr_path = os.path.join(args.out_dir, f"{args.category}_products.csv")

    transactions.assign(items=transactions["items"].apply(lambda xs: "|".join(xs))).to_csv(tx_path, index=False)
    products.to_csv(pr_path, index=False)

    print(f"\nDone.\n  transactions -> {tx_path}\n  products     -> {pr_path}")


if __name__ == "__main__":
    main()
