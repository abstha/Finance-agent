def spend_by_category(df):
    return df.groupby("Category")["Amount"].sum().sort_values()

def monthly_spend(df):
    monthly = df.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M")
    return monthly.groupby("Month")["Amount"].sum()