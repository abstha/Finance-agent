import json
import ollama
from loader import load_data
from Categorizer import hybrid_categorize, refine_transfer_category  # match your actual filename
from analytics import spend_by_category, monthly_spend
from anomaly import detect_anomalies

TOOLS = {
    "spend_by_category": "Total spending grouped by category. No arguments needed.",
    "monthly_spend": "Total spending grouped by month. No arguments needed.",
    "detect_anomalies": "Returns unusual/outlier transactions. No arguments needed.",
    "filter_by_category_and_month": "Spending for a specific category and/or month. Args: category (optional), month (optional, format YYYY-MM)."
}

def resolve_category(raw_category, valid_categories):
    raw_lower = raw_category.lower()
    for cat in valid_categories:
        if raw_lower in cat.lower() or cat.lower() in raw_lower:
            return cat
    return None  # no match found

def execute_plan(plan: list, df, valid_categories):
    # results is now a LIST, not a dict — so repeated calls to the same
    # tool (e.g. filter_by_category_and_month for 4 different months)
    # each keep their own entry instead of overwriting each other.
    results = []

    for step in plan:
        tool = step.get("tool")
        args = step.get("args", {})

        if tool == "spend_by_category":
            results.append({"tool": tool, "args": args, "result": spend_by_category(df)})

        elif tool == "monthly_spend":
            results.append({"tool": tool, "args": args, "result": monthly_spend(df)})

        elif tool == "detect_anomalies":
            results.append({"tool": tool, "args": args, "result": detect_anomalies(df)})

        elif tool == "filter_by_category_and_month":
            raw_category = args.get("category")
            month = args.get("month")

            # Skip calls that give us nothing to actually filter on —
            # prevents the "7 empty filter calls" mess from before.
            if not raw_category and not month:
                print(f"Skipping filter_by_category_and_month — no category or month given")
                continue

            filtered = df
            if raw_category:
                resolved = resolve_category(raw_category, valid_categories)
                if resolved:
                    filtered = filtered[filtered["Category"] == resolved]
                else:
                    print(f"Could not resolve category: '{raw_category}' — skipping category filter")
            if month:
                filtered = filtered[filtered["Date"].dt.to_period("M").astype(str) == month]

            results.append({"tool": tool, "args": args, "result": filtered["Amount"].sum()})

    return results

def plan_query(question: str) -> list:
    VALID_CATEGORIES = ["Groceries", "Food and dining", "Transport", "Subscriptions",
                     "Rent", "Shopping", "Fees", "Income", "Transfer In", "Transfer Out", "Education"]

    prompt = f"""You are a finance assistant planning which tools to call to answer a question.

Available tools:
{json.dumps(TOOLS, indent=2)}

User question: "{question}"

Reply with ONLY a JSON array of tool calls needed, in this exact format:
[{{"tool": "tool_name", "args": {{}}}}]

Reply with ONLY the JSON array, nothing else.
If a category argument is needed, it MUST be exactly one of: {VALID_CATEGORIES}
"""

    response = ollama.chat(
        model='phi3:mini',
        messages=[{'role': 'user', 'content': prompt}],
        # Stops generation once the model starts drifting into hallucinated
        # continuations (echoing "User question:" back, inventing new prompts, etc.)
        options={'stop': ['\n\n', 'User question', 'User:']}
    )
    raw = response["message"]["content"].strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Could not parse LLM output as JSON: {raw}")
        return []

def generate_answer(question: str, results: list) -> str:
    prompt = f"""Answer the user's question in one clear, concise response based on this data.
List every relevant data point — do not summarize or drop any.

Question: "{question}"
Data: {results}

Give a natural answer."""

    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}])
    return response["message"]["content"].strip()

if __name__ == "__main__":
    df = load_data('./data/state.csv')
    df["Category"] = df["Description"].apply(hybrid_categorize)
    df["Category"] = df.apply(refine_transfer_category, axis=1)

    VALID_CATEGORIES = ["Groceries", "Food and dining", "Transport", "Subscriptions",
                     "Rent", "Shopping", "Fees", "Income", "Transfer In", "Transfer Out", "Education"]

    questions = [
        "How much did I spend on food?",
        "What's my spending by category?",
        "Show me anything unusual in my transactions",
        "How much did I spend in July?",
        "Compare my food spending across months"
    ]

    for question in questions:
        print(f"\n--- Question: {question} ---")
        plan = plan_query(question)
        print("Plan:", plan)
        results = execute_plan(plan, df, VALID_CATEGORIES)
        print("Results:", results)
        answer = generate_answer(question, results)
        print("Answer:", answer)