import os
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT DATASET
# ============================================================

from services.search_service import df


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

QUERY_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "query_results.csv"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "ground_truth_review.csv"
)


# ============================================================
# PRODUCT ID → PRODUCT
# ============================================================

product_lookup = {}

for _, row in df.iterrows():

    product_lookup[
        str(row["product_id"])
    ] = row


# ============================================================
# LOAD PREVIOUS BENCHMARK
# ============================================================

if not os.path.exists(
    QUERY_RESULTS_PATH
):

    raise FileNotFoundError(
        "query_results.csv was not found. "
        "Run evaluate_search.py first."
    )


results_df = pd.read_csv(
    QUERY_RESULTS_PATH
)


# ============================================================
# BUILD REVIEW SET
# ============================================================

review_rows = []


grouped = results_df.groupby(
    [
        "query_id",
        "query",
    ]
)


for (
    query_id,
    query
), group in grouped:

    # --------------------------------------------------------
    # Collect union of products returned by all methods
    # --------------------------------------------------------

    candidate_ids = []

    for value in group[
        "retrieved_product_ids"
    ]:

        if pd.isna(value):
            continue

        ids = str(value).split(";")

        for product_id in ids:

            product_id = product_id.strip()

            if (
                product_id
                and product_id
                not in candidate_ids
            ):

                candidate_ids.append(
                    product_id
                )


    # --------------------------------------------------------
    # Create review row for every candidate
    # --------------------------------------------------------

    for rank, product_id in enumerate(
        candidate_ids,
        start=1
    ):

        product = product_lookup.get(
            product_id
        )


        if product is None:
            continue


        review_rows.append({

            "query_id":
                query_id,

            "query":
                query,

            "candidate_rank":
                rank,

            "product_id":
                product_id,

            "product_name":
                str(
                    product[
                        "product_name"
                    ]
                ),

            "category":
                str(
                    product[
                        "category"
                    ]
                ),

            "subcategory":
                str(
                    product[
                        "subcategory"
                    ]
                ),

            "brand":
                str(
                    product[
                        "brand"
                    ]
                ),

            "price":
                float(
                    product[
                        "price"
                    ]
                ),

            "rating":
                float(
                    product[
                        "rating"
                    ]
                ),

            "description":
                str(
                    product[
                        "description"
                    ]
                ),

            "features":
                str(
                    product[
                        "features"
                    ]
                ),

            "tags":
                str(
                    product[
                        "tags"
                    ]
                ),

            "use_cases":
                str(
                    product[
                        "use_cases"
                    ]
                ),

            # ------------------------------------------------
            # Human reviewer fills this
            # ------------------------------------------------

            "relevance":
                "",

            "reviewer_notes":
                "",

        })


# ============================================================
# SAVE
# ============================================================

review_df = pd.DataFrame(
    review_rows
)


review_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH REVIEW SET")
print("=" * 70)
print()

print(
    f"Queries: "
    f"{review_df['query_id'].nunique()}"
)

print(
    f"Candidate products: "
    f"{len(review_df)}"
)

print(
    f"Output: "
    f"{OUTPUT_PATH}"
)

print()

print(
    "Relevance labels:"
)

print(
    "  0 = Not relevant"
)

print(
    "  1 = Somewhat relevant"
)

print(
    "  2 = Highly relevant"
)

print()

print(
    "Open the CSV and fill the "
    "'relevance' column."
)

print()

print("=" * 70)