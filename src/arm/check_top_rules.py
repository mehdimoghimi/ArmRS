import sys
import pandas as pd

category = sys.argv[1] if len(sys.argv) > 1 else "Cell_Phones_and_Accessories"
rules = pd.read_csv(f"data/processed/{category}_rules.csv").sample(15, random_state=0)
products = pd.read_csv(f"data/processed/{category}_products.csv").set_index("parent_asin")

def info(asin):
    if asin in products.index:
        row = products.loc[asin]
        return f"{str(row['title'])[:60]} | ${row['price']}"
    return "(بدون متادیتا)"

for _, r in rules.iterrows():
    print(f"count={r['count']}  lift={r['lift']:.0f}")
    print(f"   A: {info(r['antecedents'])}")
    print(f"   B: {info(r['consequents'])}\n")
