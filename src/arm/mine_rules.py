"""
Phase 3 — استخراج قوانین انجمنی از تراکنش‌های فاز ۲

دو موتور:
  --engine pairs   (پیش‌فرض) قوانین یک‌به‌یک (A -> B) با شمارش هم‌رخدادی جفت‌ها.
                   معادل Apriori در سطح ۲-itemset است، ولی روی داده‌های خیلی پراکنده
                   (مثل Amazon با ده‌ها هزار محصول) سریع و کم‌مصرف کار می‌کنه.
  --engine mlxtend Apriori کامل (قوانین چندمحصولی). فقط وقتی مفیده که کاربر/تراکنش
                   زیاد و آستانه‌ها بالا باشه.

ورودی : data/processed/{CATEGORY}_transactions.csv   (خروجی فاز ۲)
خروجی : data/processed/{CATEGORY}_rules.csv
        ستون‌ها: antecedents, consequents, support, confidence, lift, count

مثال:
    python src/arm/mine_rules.py --category Cell_Phones_and_Accessories --min-count 5
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

COLS = ["antecedents", "consequents", "support", "confidence", "lift", "count"]


def load_transactions(processed_dir, category):
    path = os.path.join(processed_dir, f"{category}_transactions.csv")
    tx = pd.read_csv(path)["items"].str.split("|").tolist()
    print(f"[load] {len(tx):,} transactions from {path}")
    return tx


def mine_pairs(tx, min_count, min_item_count, max_items_per_user, min_conf, min_lift, seed=0):
    n_tx = len(tx)
    rng = np.random.default_rng(seed)

    item_freq = Counter(a for items in tx for a in set(items))
    kept = sorted(a for a, c in item_freq.items() if c >= min_item_count)
    idx = {a: i for i, a in enumerate(kept)}
    print(f"[items] {len(item_freq):,} unique items -> {len(kept):,} appear in >= {min_item_count} transactions")

    rows, cols, capped = [], [], 0
    for u, items in enumerate(tx):
        ids = [idx[a] for a in set(items) if a in idx]
        if len(ids) > max_items_per_user:
            ids = rng.choice(ids, size=max_items_per_user, replace=False).tolist()
            capped += 1
        rows.extend([u] * len(ids))
        cols.extend(ids)
    print(f"[cap] {capped:,} users had > {max_items_per_user} items and were randomly capped")

    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_tx, len(kept)))
    item_count = np.asarray(X.sum(axis=0)).ravel()

    C = (X.T @ X).tocoo()
    m = (C.row != C.col) & (C.data >= min_count)
    a, b, c = C.row[m], C.col[m], C.data[m].astype(np.int64)
    conf = c / item_count[a]
    lift = conf / (item_count[b] / n_tx)
    print(f"[pairs] {len(c):,} ordered item pairs co-occur in >= {min_count} transactions")

    keep = (conf >= min_conf) & (lift >= min_lift)
    names = np.array(kept)
    out = pd.DataFrame({
        "antecedents": names[a[keep]], "consequents": names[b[keep]],
        "support": c[keep] / n_tx, "confidence": conf[keep], "lift": lift[keep], "count": c[keep],
    })
    return out.sort_values(["lift", "confidence"], ascending=False)[COLS]


def mine_mlxtend(tx, min_support, min_conf, min_lift):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    arr = te.fit(tx).transform(tx, sparse=True)
    onehot = pd.DataFrame.sparse.from_spmatrix(arr, columns=te.columns_)
    fi = apriori(onehot, min_support=min_support, use_colnames=True)
    sizes = fi["itemsets"].apply(len).value_counts().sort_index().to_dict()
    print(f"[apriori] {len(fi):,} frequent itemsets; size distribution: {sizes}")
    if sizes.get(2, 0) + sum(v for k, v in sizes.items() if k > 2) == 0:
        print("[WARNING] only single items are frequent -> no rules possible. "
              "Lower min_support, add more users in Phase 2, or use --engine pairs.")
        return pd.DataFrame(columns=COLS)
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[r["lift"] >= min_lift].copy()
    r["antecedents"] = r["antecedents"].apply(lambda x: "|".join(sorted(x)))
    r["consequents"] = r["consequents"].apply(lambda x: "|".join(sorted(x)))
    r["count"] = (r["support"] * len(tx)).round().astype(int)
    return r.sort_values(["lift", "confidence"], ascending=False)[COLS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--engine", choices=["pairs", "mlxtend"], default="pairs")
    p.add_argument("--min-count", type=int, default=5, help="pairs: حداقل تعداد کاربرانی که هر دو محصول رو دارن")
    p.add_argument("--min-item-count", type=int, default=5, help="pairs: محصولاتی با تعداد کمتر نادیده گرفته می‌شن")
    p.add_argument("--max-items-per-user", type=int, default=100)
    p.add_argument("--min-support", type=float, default=0.005, help="فقط برای engine=mlxtend")
    p.add_argument("--min-confidence", type=float, default=0.05)
    p.add_argument("--min-lift", type=float, default=1.0)
    args = p.parse_args()

    tx = load_transactions(args.processed_dir, args.category)
    if args.engine == "pairs":
        rules = mine_pairs(tx, args.min_count, args.min_item_count, args.max_items_per_user,
                           args.min_confidence, args.min_lift)
    else:
        rules = mine_mlxtend(tx, args.min_support, args.min_confidence, args.min_lift)

    out = os.path.join(args.processed_dir, f"{args.category}_rules.csv")
    rules.to_csv(out, index=False)
    print(f"\nDone. {len(rules):,} rules -> {out}")
    if not rules.empty:
        print(rules.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
