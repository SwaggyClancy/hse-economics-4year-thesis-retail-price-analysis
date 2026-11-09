import json
from pathlib import Path
import pandas as pd

KEY = ["chain", "storeId", "productId"]
DAY = KEY + ["logicalDate"]

def load_prices(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, files = [], []
    for path in sorted(root.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or "snapshots" not in document:
            continue
        snapshots = document["snapshots"]
        source = path.relative_to(root).as_posix()
        files.append({"source": source, "rows": len(snapshots)})
        for item in snapshots:
            row = dict(item, source=source)
            for field in ("regularPrice", "discountPrice"):
                money = row.pop(field, None)
                if money is not None and (money["currency"], money["minorUnitScale"]) != ("RUB", 2):
                    raise ValueError(f"Unexpected money format in {source}")
                row[field + "Kopecks"] = money["amountMinor"] if money else None
            rows.append(row)
    if not rows:
        raise ValueError(f"No product snapshots in {root}")
    data = pd.DataFrame(rows)
    data["collectedAt"] = pd.to_datetime(data["collectedAt"], utc=True, format="ISO8601")
    data["logicalDate"] = pd.to_datetime(data["logicalDate"], format="%Y-%m-%d")
    for field in ("regularPriceKopecks", "discountPriceKopecks"):
        data[field] = data[field].astype("Int64")
    return data, pd.DataFrame(files)

def daily_panel(data: pd.DataFrame) -> pd.DataFrame:
    latest = data.groupby(DAY)["collectedAt"].transform("max")
    candidates = data.loc[data["collectedAt"].eq(latest)].copy()
    fields = ["regularPriceKopecks", "discountPriceKopecks", "available"]
    conflicts = candidates.groupby(DAY)[fields].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting latest snapshots: {conflicts.sum()}")
    panel = candidates.drop_duplicates(DAY).sort_values(DAY).copy()
    regular = panel["regularPriceKopecks"].where(panel["regularPriceKopecks"].gt(0))
    discount = panel["discountPriceKopecks"]
    panel["isPromo"] = (discount.gt(0) & discount.lt(regular)).fillna(False).where(regular.notna())
    panel["effectiveKopecks"] = regular.mask(panel["isPromo"].fillna(False), discount)
    return panel

def price_changes(panel: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    result = panel.copy()
    groups = result.groupby(KEY)
    consecutive = groups["logicalDate"].diff().eq(pd.Timedelta(days=1))
    for label, field in (("Regular", "regularPriceKopecks"), ("Effective", "effectiveKopecks")):
        price = result[field].where(result[field].gt(0))
        previous = price.groupby([result[key] for key in KEY]).shift()
        change = (price / previous - 1).where(consecutive)
        result["delta" + label] = change
        result["changed" + label] = change.abs().gt(threshold).where(change.notna())
    return result

def attach_product_links(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    links = pd.read_csv(path, dtype="string")
    columns = ["chain", "productId", "canonicalProductId"]
    links = links[columns]
    if links.isna().any().any():
        raise ValueError("Product links contain empty identifiers")
    return data.merge(links, on=columns[:2], how="left", validate="many_to_one")
