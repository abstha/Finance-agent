# Personal Finance Agent

This is a locally run, personal finance analyzer agent I built to learn basics of tool calling and agentic AI. You can upload a bank statement in CSV format and it categorizes every transaction, tracks and creates spending analytic and anomalies, and answers natural language questions such as ("How much did I spend at KFC?", "How much did I pay for subscriptions in August?"). The agent creates and LLM plan and executes calls against its own tool set all while running entirely on device with no data sent to the cloud.

Built as a hands-on follow-up to IBM's Agentic AI course (Coursera), applying the concepts to a real, personally useful problem.

## Why this project

Most "AI finance app" demos wrap a single LLM call around a spreadsheet. This one is closer to how agentic systems actually behave in practice with a small local model that is genuinely unreliable at structured output, a hybrid architecture that compensates for that unreliability, and a tool-calling layer the model has to plan against rather than a fixed script. Getting it to work well against **messy, real bank data** and not a clean synthetic demo set created some interesting engineering problems.

## Architecture

```
Bank statement CSV
       │
       ▼
┌─────────────────┐
│   loader.py      │  Parses raw export → typed DataFrame
│                  │  (dates, signed amounts, currency cleanup)
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│   categorizer.py             │  Hybrid categorization:
│   1. Rule-based keywords     │  1. fast, free, deterministic
│   2. LLM fallback (Phi-3)    │  2. handles ambiguous/novel merchants
│   3. Output validation       │  3. constrains LLM to a fixed category set
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────┐   ┌──────────────────┐
│  analytics.py    │   │   anomaly.py      │
│  spend_by_category│  │  z-score outlier  │
│  monthly_spend    │  │  detection         │
│  filter_by_merchant│ │                    │
└────────┬─────────┘   └────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   agent.py             │  Natural-language query layer:
         │   1. plan_query()      │  LLM plans which tool(s) to call
         │   2. execute_plan()    │  Runs the plan against real data
         │   3. generate_answer() │  LLM writes a plain-English answer
         └───────────┬─────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   app.py (Streamlit)   │  Dashboard: overview charts,
         │                        │  anomalies, ask-a-question, raw data
         └───────────────────────┘
```

All LLM calls run against **Phi-3-mini (3.8B)** via **Ollama**, fully local — no API keys, no cost per query, no transaction data ever sent to a third party.

## Features

- **Hybrid transaction categorization** — rule-based keyword matching first, LLM fallback for anything ambiguous, with output validated against a fixed category list so the model can't silently invent or fragment categories.
- **Spending analytics** — totals by category, trend by month, all computed with pandas.
- **Anomaly detection** — z-score based outlier flagging (correctly identified a one-off tuition payment as a statistical anomaly distinct from regular spending, with no hand-coded rule for it).
- **Agentic query layer** — ask a question in plain English; an LLM plans which internal tool(s) to call (category filter, month filter, merchant search, anomaly check — including multiple calls for multi-part questions like month-by-month comparisons), the plan is executed against the real data, and the result is turned back into a natural-language answer.
- **Streamlit dashboard** — upload a statement, browse spending by category/month, inspect flagged anomalies, and ask the agent questions directly, all in one UI.

## Tech stack

Python · pandas · Ollama (Phi-3-mini) · Streamlit

## Setup

```bash
git clone https://github.com/abstha/Finance-agent.git
cd finance-agent
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt

# Install Ollama (https://ollama.com), then pull the model:
ollama pull phi3:mini

streamlit run app.py
```

Upload a bank statement CSV or PDF in the sidebar to get started. No real data is included in this repo — see `.gitignore`.

## Example queries

| Question | What the agent does |
|---|---|
| "How much did I spend on food?" | Filters by category, sums |
| "Compare my food spending across months" | Plans one filter call per month present in the data, returns all of them |
| "How much did I spend at KFC?" | Searches transaction descriptions by merchant keyword |
| "Show me anything unusual in my transactions" | Runs z-score anomaly detection |

## Engineering challenges and how they were handled

While building this agent with **real** bank csv and pdf statements, several problems worth documenting surfaced which I feel are worth mentioning here.

- **LLM category drift** — without constraints, the categorizer would invent near-duplicate categories ("Food and Dining" vs "food and dining", or entirely new ones like "Overdraft Fees"), silently fragmenting the analytics. This was fixed by validating every LLM output against a fixed category list and falling back to "Uncategorized" on any mismatch.
- **Small-model unreliability in structured output** — phi3:mini frequently wrapped JSON responses in markdown which occasionally produced malformed JSON (mismatched brackets). In early testing the model hallucinated continuations of the prompt template itself. This was addressed with fence-stripping, regex-based JSON extraction, an Ollama `stop` sequence to cut off generation before it drifts, and a separate fallback (empty plan) on unparseable output rather than crashing.
- **Date/year hallucination** — when asked to filter by month, the model would sometimes invent a plausible-but-wrong year (e.g. 2022 for data that only spans 2026). This was fixed by explicitly passing the list of months actually present in the uploaded data into the planning prompt, so the model can only select real options.
- **Silent result overwriting** — an early version of the multi-tool executor stored results in a dict keyed by tool name, so a multi-month comparison (the same tool called several times) silently lost all but the last result. Fixed by switching to a list of tagged results.
- **A single large transaction distorting aggregates** — a one-off ~$13,800 tuition payment dominated category totals and made the "Uncategorized" bucket look far larger than it was. This became the motivating test case for anomaly detection rather than something to filter out by hand.

## Known limitations

- Runs on a 3.8B local model by design (privacy + zero cost); a larger or cloud-hosted model would plan more reliably, at the cost of privacy and per-query cost.
- The agent's query understanding is bounded by its tool set — questions requiring a tool that doesn't exist yet (e.g. very open-ended comparisons) may produce an incomplete or unhelpful plan.
- Multi-step planning is not fully deterministic; the same question can occasionally produce a different (but usually still correct) plan across runs.

## Possible future work

- Broaden the tool set (e.g. spending trend by merchant, budget vs. actual comparisons).
- Swap z-score anomaly detection for a more robust method (e.g. median absolute deviation), since a single extreme outlier currently inflates the standard deviation used to detect others.