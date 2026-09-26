import os
import math
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

QUERY_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "query_results.csv"
)

GROUND_TRUTH_PATH = os.path.join(
    RESULTS_DIR,
    "ground_truth_review.csv"
)

OUTPUT_SUMMARY = os.path.join(
    RESULTS_DIR,
    "human_ground_truth_summary.csv"
)

OUTPUT_QUERY = os.path.join(
    RESULTS_DIR,
    "human_ground_truth_query_results.csv"
)

if not os.path.exists(QUERY_RESULTS_PATH):
    raise FileNotFoundError(
        "query_results.csv not found. Run evaluate_search.py first."
    )

if not os.path.exists(GROUND_TRUTH_PATH):
    raise FileNotFoundError(
        "ground_truth_review.csv not found. "
        "Fill the relevance column before running this script."
    )

results_df = pd.read_csv(QUERY_RESULTS_PATH)
truth_df = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep=None,
    engine="python"
)
truth_df.columns = [
    "query_id",
    "query",
    "candidate_rank",
    "product_id",
    "product_name",
    "category",
    "subcategory",
    "brand",
    "price",
    "rating",
    "description",
    "features",
    "tags",
    "use_cases",
    "relevance",
    "reviewer_notes"
]

if "relevance" not in truth_df.columns:
    raise ValueError("ground_truth_review.csv must contain a relevance column.")

truth_df["relevance"] = pd.to_numeric(
    truth_df["relevance"],
    errors="coerce"
)

if truth_df["relevance"].isna().any():
    missing = int(truth_df["relevance"].isna().sum())
    raise ValueError(
        f"{missing} relevance labels are missing. "
        "Every candidate must have 0, 1, or 2."
    )

if not truth_df["relevance"].isin([0, 1, 2]).all():
    raise ValueError(
        "Relevance values must only be 0, 1, or 2."
    )

label_lookup = {
    (str(row["query_id"]), str(row["product_id"])): int(row["relevance"])
    for _, row in truth_df.iterrows()
}


def dcg_at_k(relevances, k=5):
    score = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        score += (2 ** rel - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(relevances, ideal_relevances, k=5):
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(
        sorted(ideal_relevances, reverse=True),
        k
    )
    if ideal == 0:
        return 0.0
    return actual / ideal


query_rows = []

for (query_id, query), group in results_df.groupby(
    ["query_id", "query"],
    sort=False
):
    # The existing benchmark stores one row per method/query.
    # retrieved_product_ids is a semicolon-separated list.
    all_judged = truth_df[
        truth_df["query_id"].astype(str) == str(query_id)
    ]

    ideal_relevances = all_judged["relevance"].tolist()

    for method, method_group in group.groupby("method", sort=False):
        row = method_group.iloc[0]

        raw_ids = str(
            row["retrieved_product_ids"]
        )

        retrieved_ids = [
            x.strip()
            for x in raw_ids.split(";")
            if x.strip()
        ][:5]

        relevances = [
            label_lookup.get(
                (str(query_id), product_id),
                0
            )
            for product_id in retrieved_ids
        ]

        # Binary relevance for Precision/Recall/MRR.
        binary = [
            1 if rel > 0 else 0
            for rel in relevances
        ]

        relevant_retrieved = sum(binary)

        # Total judged relevant products for this query.
        total_relevant = int(
            (all_judged["relevance"] > 0).sum()
        )

        precision = (
            relevant_retrieved / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        recall = (
            relevant_retrieved / total_relevant
            if total_relevant > 0 else 0.0
        )

        reciprocal_rank = 0.0
        for rank, rel in enumerate(relevances, start=1):
            if rel > 0:
                reciprocal_rank = 1.0 / rank
                break

        ndcg = ndcg_at_k(
            relevances,
            ideal_relevances,
            k=5
        )

        query_rows.append({
            "query_id": query_id,
            "query": query,
            "method": method,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "mrr": reciprocal_rank,
            "ndcg_at_5": ndcg,
            "latency_ms": float(row["latency_ms"]),
            "relevant_retrieved": relevant_retrieved,
            "total_judged_relevant": total_relevant,
            "retrieved_product_ids": ";".join(retrieved_ids),
            "relevance_labels": ";".join(
                str(x) for x in relevances
            )
        })

query_metrics = pd.DataFrame(query_rows)

summary = (
    query_metrics
    .groupby("method", as_index=False)
    .agg(
        Precision_at_5=("precision_at_5", "mean"),
        Recall_at_5=("recall_at_5", "mean"),
        MRR=("mrr", "mean"),
        NDCG_at_5=("ndcg_at_5", "mean"),
        Avg_Latency_ms=("latency_ms", "mean")
    )
)

summary = summary.sort_values(
    "NDCG_at_5",
    ascending=False
)

query_metrics.to_csv(
    OUTPUT_QUERY,
    index=False
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print()
print("=" * 78)
print("HUMAN-GROUND-TRUTH SEARCH EVALUATION")
print("=" * 78)
print()
print(summary.to_string(index=False))
print()
print(f"Ground truth: {GROUND_TRUTH_PATH}")
print(f"Query metrics: {OUTPUT_QUERY}")
print(f"Summary: {OUTPUT_SUMMARY}")
print()
print("=" * 78)
print("EVALUATION COMPLETE")
print("=" * 78)
