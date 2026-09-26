import os
import re
import sys
import time
import math
import json

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
# EXISTING VANTOS SEARCH COMPONENTS
# ============================================================

from services.search_service import (
    client,
    model,
    COLLECTION_NAME,
    df,
    extract_category_from_query,
    extract_max_price_from_query,
    search_products,
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    {
        "query_id": "Q01",
        "query":
            "lightweight laptop for software development under 50000",
        "category": "laptops",
        "max_price": 50000,
        "required_terms": [
            "lightweight",
            "software development"
        ]
    },

    {
        "query_id": "Q02",
        "query":
            "developer laptop for programming under 80000",
        "category": "laptops",
        "max_price": 80000,
        "required_terms": [
            "software development",
            "programming"
        ]
    },

    {
        "query_id": "Q03",
        "query":
            "student laptop for college under 60000",
        "category": "laptops",
        "max_price": 60000,
        "required_terms": [
            "college",
            "study"
        ]
    },

    {
        "query_id": "Q04",
        "query":
            "gaming laptop for performance under 100000",
        "category": "laptops",
        "max_price": 100000,
        "required_terms": [
            "gaming",
            "performance"
        ]
    },

    {
        "query_id": "Q05",
        "query":
            "premium laptop for productivity",
        "category": "laptops",
        "max_price": None,
        "required_terms": [
            "productivity"
        ]
    },

    {
        "query_id": "Q06",
        "query":
            "best camera phone under 25000",
        "category": "mobiles",
        "max_price": 25000,
        "required_terms": [
            "camera",
            "photography"
        ]
    },

    {
        "query_id": "Q07",
        "query":
            "gaming smartphone under 30000",
        "category": "mobiles",
        "max_price": 30000,
        "required_terms": [
            "gaming",
            "performance"
        ]
    },

    {
        "query_id": "Q08",
        "query":
            "phone for photography around 50000",
        "category": "mobiles",
        "max_price": 50000,
        "required_terms": [
            "photography"
        ]
    },

    {
        "query_id": "Q09",
        "query":
            "battery focused phone for travel under 20000",
        "category": "mobiles",
        "max_price": 20000,
        "required_terms": [
            "battery",
            "travel"
        ]
    },

    {
        "query_id": "Q10",
        "query":
            "blue formal shirt for office",
        "category": "fashion",
        "max_price": None,
        "required_terms": [
            "blue",
            "formal",
            "office"
        ]
    },

    {
        "query_id": "Q11",
        "query":
            "black casual shirt for everyday use",
        "category": "fashion",
        "max_price": None,
        "required_terms": [
            "black",
            "casual"
        ]
    },

    {
        "query_id": "Q12",
        "query":
            "running shoes for fitness under 5000",
        "category": "shoes",
        "max_price": 5000,
        "required_terms": [
            "running",
            "fitness"
        ]
    },

    {
        "query_id": "Q13",
        "query":
            "formal shoes for office under 6000",
        "category": "shoes",
        "max_price": 6000,
        "required_terms": [
            "formal",
            "office"
        ]
    },

    {
        "query_id": "Q14",
        "query":
            "waterproof laptop bag for office",
        "category": "accessories",
        "max_price": None,
        "required_terms": [
            "water-resistant",
            "laptop",
            "office"
        ]
    },

    {
        "query_id": "Q15",
        "query":
            "power bank for travel under 3000",
        "category": "accessories",
        "max_price": 3000,
        "required_terms": [
            "power bank",
            "travel"
        ]
    },

    {
        "query_id": "Q16",
        "query":
            "USB C cable for charging under 1500",
        "category": "accessories",
        "max_price": 1500,
        "required_terms": [
            "usb-c",
            "charging"
        ]
    },

    {
        "query_id": "Q17",
        "query":
            "smart home electronics for office under 25000",
        "category": "electronics",
        "max_price": 25000,
        "required_terms": [
            "smart",
            "office"
        ]
    },

    {
        "query_id": "Q18",
        "query":
            "digital camera for photography under 100000",
        "category": "electronics",
        "max_price": 100000,
        "required_terms": [
            "camera",
            "photography"
        ]
    },

]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace(
        "usb-c",
        "usb c"
    )

    text = re.sub(
        r"[^a-z0-9₹ ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def product_searchable_text(row):

    fields = [

        row.get("product_name", ""),

        row.get("category", ""),

        row.get("subcategory", ""),

        row.get("description", ""),

        row.get("features", ""),

        row.get("tags", ""),

        row.get("use_cases", ""),

    ]

    return normalize_text(
        " ".join(
            str(field)
            for field in fields
        )
    )


# ============================================================
# HARD QUERY FILTERS
# ============================================================

def query_filters(query):

    category = (
        extract_category_from_query(
            query
        )
    )

    max_price = (
        extract_max_price_from_query(
            query
        )
    )

    return (
        category,
        max_price
    )


# ============================================================
# KEYWORD SEARCH BASELINE
# ============================================================

def keyword_search(
    query,
    top_k=5
):

    category, max_price = (
        query_filters(query)
    )

    query_terms = [
        term
        for term in normalize_text(
            query
        ).split()
        if len(term) > 1
    ]

    scored_products = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # Category constraint
        # ----------------------------------------------------

        if category:

            if (
                str(row["category"]).lower()
                != category.lower()
            ):
                continue


        # ----------------------------------------------------
        # Price constraint
        # ----------------------------------------------------

        if max_price is not None:

            if (
                float(row["price"])
                > max_price
            ):
                continue


        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        if (
            str(
                row["availability"]
            ).lower()
            != "in_stock"
        ):
            continue


        searchable_text = (
            product_searchable_text(
                row
            )
        )


        score = 0


        for term in query_terms:

            if term in searchable_text:

                score += 1


        if score > 0:

            scored_products.append(
                (
                    score,
                    row
                )
            )


    # Highest lexical match first

    scored_products.sort(
        key=lambda item: item[0],
        reverse=True
    )


    return [
        row["product_id"]
        for _, row
        in scored_products[:top_k]
    ]


# ============================================================
# PURE SEMANTIC SEARCH
#
# This is deliberately different from the Qwen pipeline.
# It performs:
#
# Query → embedding → Qdrant
#
# No Qwen intent extraction.
# ============================================================

def semantic_search(
    query,
    top_k=5
):

    category, max_price = (
        query_filters(query)
    )


    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    conditions = []


    if category:

        from qdrant_client.models import (
            FieldCondition,
            MatchValue,
        )

        conditions.append(

            FieldCondition(
                key="category",
                match=MatchValue(
                    value=category
                )
            )

        )


    if max_price is not None:

        from qdrant_client.models import (
            FieldCondition,
            Range,
        )

        conditions.append(

            FieldCondition(
                key="price",
                range=Range(
                    lte=max_price
                )
            )

        )


    from qdrant_client.models import (
        FieldCondition,
        MatchValue,
    )

    conditions.append(

        FieldCondition(
            key="availability",
            match=MatchValue(
                value="in_stock"
            )
        )

    )


    from qdrant_client.models import (
        Filter,
    )

    search_filter = Filter(
        must=conditions
    )


    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        query_filter=search_filter,

        limit=top_k

    ).points


    return [

        str(
            result.payload[
                "product_id"
            ]
        )

        for result in results

    ]


# ============================================================
# TWO-PASS SEARCH
#
# Uses the actual VANTOS implementation.
# ============================================================

def two_pass_search(
    query,
    top_k=5
):

    result = search_products(
        query,
        top_k
    )


    return [

        str(
            result.payload[
                "product_id"
            ]
        )

        for result
        in result["results"]

    ]


# ============================================================
# GROUND TRUTH
#
# Because this is a synthetic catalog, relevance is defined
# using explicit structured product metadata.
# ============================================================

def build_ground_truth():

    ground_truth = []


    for test in TEST_QUERIES:

        relevant = []


        for _, row in df.iterrows():

            # Category must match

            if (
                str(row["category"]).lower()
                != test["category"].lower()
            ):
                continue


            # Budget must match

            if (
                test["max_price"]
                is not None
            ):

                if (
                    float(row["price"])
                    > test["max_price"]
                ):
                    continue


            # Only available products

            if (
                str(
                    row["availability"]
                ).lower()
                != "in_stock"
            ):
                continue


            searchable_text = (
                product_searchable_text(
                    row
                )
            )


            # Count semantic/business requirements

            matched_terms = 0


            for term in test[
                "required_terms"
            ]:

                normalized_term = (
                    normalize_text(term)
                )


                if normalized_term in searchable_text:

                    matched_terms += 1


            # All required concepts must match

            if (
                matched_terms
                == len(
                    test[
                        "required_terms"
                    ]
                )
            ):

                relevant.append(
                    str(
                        row["product_id"]
                    )
                )


        ground_truth.append({

            "query_id":
                test["query_id"],

            "query":
                test["query"],

            "category":
                test["category"],

            "max_price":
                test["max_price"],

            "relevant_product_ids":
                ";".join(
                    relevant
                ),

            "relevant_count":
                len(relevant),

        })


    return pd.DataFrame(
        ground_truth
    )


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k=5
):

    retrieved = retrieved[:k]

    if not retrieved:

        return 0.0


    relevant_set = set(
        relevant
    )


    hits = sum(
        1
        for product_id
        in retrieved
        if product_id
        in relevant_set
    )


    return hits / len(
        retrieved
    )


def recall_at_k(
    retrieved,
    relevant,
    k=5
):

    relevant_set = set(
        relevant
    )

    if not relevant_set:

        return 0.0


    retrieved = retrieved[:k]


    hits = sum(
        1
        for product_id
        in retrieved
        if product_id
        in relevant_set
    )


    return hits / len(
        relevant_set
    )


def reciprocal_rank(
    retrieved,
    relevant
):

    relevant_set = set(
        relevant
    )


    for index, product_id in enumerate(
        retrieved,
        start=1
    ):

        if product_id in relevant_set:

            return 1.0 / index


    return 0.0


def ndcg_at_k(
    retrieved,
    relevant,
    k=5
):

    relevant_set = set(
        relevant
    )


    retrieved = retrieved[:k]


    dcg = 0.0


    for index, product_id in enumerate(
        retrieved,
        start=1
    ):

        relevance = (
            1
            if product_id in relevant_set
            else 0
        )


        dcg += (
            relevance
            / math.log2(index + 1)
        )


    ideal_hits = min(
        len(relevant_set),
        k
    )


    if ideal_hits == 0:

        return 0.0


    idcg = sum(

        1
        / math.log2(index + 1)

        for index
        in range(
            1,
            ideal_hits + 1
        )

    )


    return dcg / idcg


# ============================================================
# RUN SEARCH WITH TIMING
# ============================================================

def timed_search(
    search_function,
    query
):

    start = time.perf_counter()


    results = search_function(
        query,
        TOP_K
    )


    end = time.perf_counter()


    latency_ms = (
        end - start
    ) * 1000


    return (
        results,
        latency_ms
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VANTOS SEARCH EVALUATION")
    print("=" * 70)
    print()

    print(
        f"Dataset size: {len(df)} products"
    )

    print(
        f"Test queries: {len(TEST_QUERIES)}"
    )

    print()


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth_df = (
        build_ground_truth()
    )


    ground_truth_path = os.path.join(
        RESULTS_DIR,
        "ground_truth.csv"
    )


    ground_truth_df.to_csv(
        ground_truth_path,
        index=False
    )


    print(
        f"Ground truth saved to: "
        f"{ground_truth_path}"
    )

    print()


    # --------------------------------------------------------
    # Evaluation records
    # --------------------------------------------------------

    records = []


    for index, test in enumerate(
        TEST_QUERIES,
        start=1
    ):

        query_id = test[
            "query_id"
        ]

        query = test[
            "query"
        ]


        ground_truth_row = (
            ground_truth_df[
                ground_truth_df[
                    "query_id"
                ]
                == query_id
            ]
            .iloc[0]
        )


        relevant = [

            product_id

            for product_id
            in str(
                ground_truth_row[
                    "relevant_product_ids"
                ]
            ).split(";")

            if product_id

        ]


        print(
            f"[{index:02d}/{len(TEST_QUERIES)}]"
            f" {query}"
        )


        # ====================================================
        # KEYWORD
        # ====================================================

        keyword_results, keyword_latency = (
            timed_search(
                keyword_search,
                query
            )
        )


        # ====================================================
        # SEMANTIC
        # ====================================================

        semantic_results, semantic_latency = (
            timed_search(
                semantic_search,
                query
            )
        )


        # ====================================================
        # TWO PASS
        # ====================================================

        two_pass_results, two_pass_latency = (
            timed_search(
                two_pass_search,
                query
            )
        )


        # ====================================================
        # METRICS
        # ====================================================

        methods = [

            (
                "keyword",
                keyword_results,
                keyword_latency
            ),

            (
                "semantic",
                semantic_results,
                semantic_latency
            ),

            (
                "semantic_qwen",
                two_pass_results,
                two_pass_latency
            ),

        ]


        for (
            method,
            retrieved,
            latency
        ) in methods:

            records.append({

                "query_id":
                    query_id,

                "query":
                    query,

                "method":
                    method,

                "retrieved_product_ids":
                    ";".join(
                        retrieved
                    ),

                "relevant_product_count":
                    len(relevant),

                "precision_at_5":
                    precision_at_k(
                        retrieved,
                        relevant,
                        TOP_K
                    ),

                "recall_at_5":
                    recall_at_k(
                        retrieved,
                        relevant,
                        TOP_K
                    ),

                "mrr":
                    reciprocal_rank(
                        retrieved,
                        relevant
                    ),

                "ndcg_at_5":
                    ndcg_at_k(
                        retrieved,
                        relevant,
                        TOP_K
                    ),

                "latency_ms":
                    latency,

            })


    # ========================================================
    # SAVE QUERY RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        records
    )


    results_path = os.path.join(
        RESULTS_DIR,
        "query_results.csv"
    )


    results_df.to_csv(
        results_path,
        index=False
    )


    # ========================================================
    # AGGREGATE
    # ========================================================

    summary = (

        results_df

        .groupby("method")

        .agg({

            "precision_at_5":
                "mean",

            "recall_at_5":
                "mean",

            "mrr":
                "mean",

            "ndcg_at_5":
                "mean",

            "latency_ms":
                "mean",

        })

        .reset_index()

    )


    summary = summary.rename(

        columns={

            "method":
                "Method",

            "precision_at_5":
                "Precision@5",

            "recall_at_5":
                "Recall@5",

            "mrr":
                "MRR",

            "ndcg_at_5":
                "NDCG@5",

            "latency_ms":
                "Avg Latency (ms)",

        }

    )


    summary_path = os.path.join(
        RESULTS_DIR,
        "search_benchmark_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SEARCH BENCHMARK RESULTS")
    print("=" * 70)
    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}"
        )
    )

    print()

    print(
        "Ground truth:",
        ground_truth_path
    )

    print(
        "Query results:",
        results_path
    )

    print(
        "Summary:",
        summary_path
    )

    print()

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()