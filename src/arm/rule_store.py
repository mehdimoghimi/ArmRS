"""
بازیابی و رتبه‌بندی قوانین انجمنی — دقیقاً همون retrieve_rules() که در
بخش «زنجیره فنی» پروپوزال توصیف شده. فاز ۵ این ماژول رو مستقیم import می‌کنه.
"""

import pandas as pd


def load_rule_store(processed_dir: str, category: str):
    rules = pd.read_csv(f"{processed_dir}/{category}_rules.csv")
    products = pd.read_csv(f"{processed_dir}/{category}_products.csv")
    rules["antecedents"] = rules["antecedents"].str.split("|")
    rules["consequents"] = rules["consequents"].str.split("|")
    return rules, products


def retrieve_rules(rules_df: pd.DataFrame, products_df: pd.DataFrame,
                    seed_items=None, category_keyword=None, budget_max=None, top_k=10):
    """
    seed_items       : لیست parent_asin که کاربر از قبل نشون داده (مثلاً در سناریوی Cross-selling)
    category_keyword : کلمه کلیدی دسته/نوع کاربرد که LLM از جمله کاربر استخراج کرده (مثلاً "phone case")
    budget_max        : سقف بودجه کاربر
    خروجی: لیست دیکشنری‌های محصول کاندید، به همراه متادیتای قانونی که آن‌ها را پیشنهاد داده (support/confidence/lift)
    """
    candidates = rules_df.copy()

    if seed_items:
        seed_set = set(seed_items)
        candidates = candidates[candidates["antecedents"].apply(
            lambda a: bool(seed_set & set(a)))]

    if category_keyword:
        mask = (
            products_df["category"].str.contains(category_keyword, case=False, na=False)
            | products_df["title"].str.contains(category_keyword, case=False, na=False)
        )
        valid_asins = set(products_df.loc[mask, "parent_asin"])
        candidates = candidates[candidates["consequents"].apply(
            lambda c: bool(set(c) & valid_asins))]

    candidates = candidates.sort_values(["lift", "confidence"], ascending=False)

    results, seen = [], set()
    for _, row in candidates.iterrows():
        for asin in row["consequents"]:
            if asin in seen:
                continue
            prod_rows = products_df[products_df["parent_asin"] == asin]
            if prod_rows.empty:
                continue  # متادیتا نداره -> طبق تصمیم فاز ۲، از پیشنهاد به کاربر حذف می‌شه
            prod = prod_rows.iloc[0]
            if pd.isna(prod["price"]):
                continue
            if budget_max is not None and prod["price"] > budget_max:
                continue
            results.append({
                "parent_asin": asin,
                "title": prod["title"],
                "price": float(prod["price"]),
                "brand": prod["brand"],
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            })
            seen.add(asin)
        if len(results) >= top_k:
            break
    return results[:top_k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--keyword", default=None, help="e.g. case, charger, headphone")
    parser.add_argument("--budget-max", type=float, default=None)
    args = parser.parse_args()

    rules, products = load_rule_store(args.processed_dir, args.category)
    out = retrieve_rules(rules, products, category_keyword=args.keyword, budget_max=args.budget_max)
    for r in out:
        print(f"  {r['title'][:50]:50s}  ${r['price']:.2f}  conf={r['confidence']:.2f}  lift={r['lift']:.2f}")
