def spend_by_category(df):
    return df.groupby("Category")["Amount"].sum().sort_values()

def monthly_spend(df):
    monthly = df.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M")
    return monthly.groupby("Month")["Amount"].sum()

def filter_by_merchant(df, keyword: str):
    mask = df["Description"].str.lower().str.contains(keyword.lower(), na=False)
    return df[mask]["Amount"].sum()