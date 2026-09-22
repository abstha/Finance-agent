import ollama
from loader import categorize, load_data
from analytics import spend_by_category as sp
from analytics import monthly_spend as ms
from anomaly import detect_anomalies as da

VALID_CATEGORIES = ["Groceries", "Food and dining", "Transport", "Subscriptions",
                     "Rent", "Entertainment", "Shopping", "Transfer", "Uncategorized"]

def llm_categorizer(merchant: str) -> str:
    prompt = f"""Categorize this bank transaction into exactly one of these categories: {", ".join(VALID_CATEGORIES)}

Transaction: "{merchant}"

Reply with only the category name, nothing else."""
    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}])
    result = response["message"]["content"].strip()
    return result if result in VALID_CATEGORIES else "Uncategorized"

def hybrid_categorize(merchant: str) -> str:
    result = categorize(merchant)
    if result == "Uncategorized":
        result = llm_categorizer(merchant)
    return result

def refine_transfer_category(row):
    if row["Category"] == "Transfer":
        return "Transfer In" if row["Amount"] > 0 else "Transfer Out"
    return row["Category"]

if __name__ == "__main__":
    df = load_data('./data/state.csv')
    df["Category"] = df["Description"].apply(hybrid_categorize)
    df["Category"] = df.apply(refine_transfer_category, axis=1)   # ← add this line
    print("spend by category \n")
    print(sp(df))
    print("spend by month \n")
    print(ms(df))
    print (da(df))