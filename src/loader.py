import pandas as pd

CATEGORY_RULES = {
    "Groceries": ["woolworths", "aldi", "coles", "bazaar"],
    "Shopping" : ["kmart", "big w", "mart"],
    "Food and dining": ["doordash", "pappa flock", "bamboo","kfc", "fish", "flamed", "bento", "mexican", "juice", "lounge", "cakeman", "hello india", "hjs"],
    "Transport": ["didi", "opal", "uber", "transportfornsw"],
    "Fees": ["excess interest", "overdraw fee"],
    "Income": ["direct credit", "ato"],
    "Subscriptions": ["dodo", "openai", "claude", "gemini"],
    "Rent": ["rent"],
    "Entertainment": ["game"],
    "Transfer" : ["transfer", "revolut", "rmtly"],
    "Education": ["macquarie university"],
}

def categorize(merchant: str) -> str:
    merchant_lower = merchant.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in merchant_lower for kw in keywords):
            return category
    return "Uncategorized"

def load_data(inp):
    df = pd.read_csv(
        inp,
        sep=",",
        header=None,
        names=["Date", "Amount", "Description", "Balance"]
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df

def refine_transfer_category(row):
    if row["Category"] == "Transfer":
        return "Transfer In" if row["Amount"] > 0 else "Transfer Out"
    return row["Category"]


if __name__ == "__main__":
    df = load_data('./data/state.csv')
    df["Category"] = df["Description"].apply(categorize)
    
    df["Category"] = df.apply(refine_transfer_category, axis=1)
    print(df["Category"].value_counts())
    print(df[df["Category"] == "Uncategorized"]["Description"].unique())
    print(df[df["Category"] == "Uncategorized"][["Description", "Amount"]].sort_values("Amount"))