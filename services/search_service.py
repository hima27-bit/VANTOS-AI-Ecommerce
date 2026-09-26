import json
import re
import pandas as pd
import requests

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)


# ============================================================
# CONFIGURATION
# ============================================================

QWEN_URL = "http://127.0.0.1:8080/v1/chat/completions"

COLLECTION_NAME = "products"

QDRANT_PATH = "./qdrant_data"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# LOAD MODELS / DATA
# ============================================================

print("Loading VANTOS search service...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

client = QdrantClient(
    path=QDRANT_PATH
)

df = pd.read_csv(
    "products.csv"
)

print("Search service loaded successfully.")


# ============================================================
# NORMALIZE INTENT
# ============================================================

def normalize_intent(intent):

    if not isinstance(intent, dict):
        intent = {}

    category = intent.get("category")

    if category:

        category = str(
            category
        ).lower().strip()

        category_map = {
            "laptop": "laptops",
            "laptops": "laptops",

            "mobile": "mobiles",
            "mobiles": "mobiles",

            "phone": "mobiles",
            "phones": "mobiles",

            "smartphone": "mobiles",
            "smartphones": "mobiles",

            "electronic": "electronics",
            "electronics": "electronics",

            "fashion": "fashion",

            "shoe": "shoes",
            "shoes": "shoes",

            "accessory": "accessories",
            "accessories": "accessories",
        }

        category = category_map.get(
            category,
            category
        )

    else:

        category = None


    max_price = intent.get(
        "max_price"
    )

    if max_price is not None:

        try:

            max_price = float(
                str(max_price)
                .replace(",", "")
                .replace("₹", "")
                .strip()
            )

        except (
            ValueError,
            TypeError
        ):

            max_price = None


    preferences = intent.get(
        "preferences",
        []
    )

    if isinstance(
        preferences,
        str
    ):

        preferences = [
            preferences
        ]

    elif not isinstance(
        preferences,
        list
    ):

        preferences = []


    # Remove duplicate preferences while preserving order
    cleaned_preferences = []

    for preference in preferences:
        preference = str(preference).strip()

        if not preference:
            continue

        if preference.lower() not in {
            str(existing).lower()
            for existing in cleaned_preferences
        }:
            cleaned_preferences.append(preference)

    return {
        "category": category,
        "intent": intent.get("intent"),
        "max_price": max_price,
        "preferences": cleaned_preferences
    }


# ============================================================
# FALLBACK CATEGORY EXTRACTION
# ============================================================

def extract_category_from_query(query):

    query_lower = query.lower()

    # -------------------------------------------------------
    # Accessories
    # Check these BEFORE mobile keywords because words like
    # "headphones" contain the substring "phone".
    # -------------------------------------------------------
    accessory_keywords = [
        "accessory",
        "accessories",
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "earbuds",
        "smartwatch",
        "smart watch",
        "watch",
        "charger",
        "charging cable",
        "power bank",
        "phone case",
        "case",
        "keyboard",
        "mouse",
        "speaker",
        "laptop bag",
        "backpack",
        "bag",
    ]

    if any(
        keyword in query_lower
        for keyword in accessory_keywords
    ):
        return "accessories"

    # -------------------------------------------------------
    # Laptops
    # -------------------------------------------------------
    laptop_keywords = [
        "laptop",
        "laptops",
        "notebook",
    ]

    if any(
        keyword in query_lower
        for keyword in laptop_keywords
    ):
        return "laptops"

    # -------------------------------------------------------
    # Mobiles
    # -------------------------------------------------------
    mobile_keywords = [
        "mobile",
        "mobiles",
        "smartphone",
        "smartphones",
        "iphone",
        "android phone",
        "android mobile",
        "cell phone",
    ]

    if any(
        keyword in query_lower
        for keyword in mobile_keywords
    ):
        return "mobiles"

    # -------------------------------------------------------
    # Electronics
    # -------------------------------------------------------
    electronics_keywords = [
        "electronics",
        "electronic",
        "television",
        "tv",
        "monitor",
        "camera",
        "printer",
        "speaker",
        "gaming console",
        "console",
    ]

    if any(
        keyword in query_lower
        for keyword in electronics_keywords
    ):
        return "electronics"

    # -------------------------------------------------------
    # Fashion
    # -------------------------------------------------------
    fashion_keywords = [
        "fashion",
        "shirt",
        "shirts",
        "t-shirt",
        "tshirt",
        "jeans",
        "dress",
        "jacket",
        "clothing",
        "formal shirt",
        "casual shirt",
    ]

    if any(
        keyword in query_lower
        for keyword in fashion_keywords
    ):
        return "fashion"

    # -------------------------------------------------------
    # Shoes
    # -------------------------------------------------------
    shoe_keywords = [
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boots",
        "running shoes",
        "formal shoes",
        "sports shoes",
    ]

    if any(
        keyword in query_lower
        for keyword in shoe_keywords
    ):
        return "shoes"

    return None

# ============================================================
# FALLBACK PRICE EXTRACTION
# ============================================================

def extract_max_price_from_query(query):

    query_lower = query.lower()

    patterns = [

        r"(?:under|below|less than|within|max(?:imum)?(?: price)?(?: of)?)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)",

        r"(?:₹|rs\.?|inr)\s*([\d,]+)\s*(?:or less|maximum|max)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query_lower
        )

        if match:

            try:

                value = float(
                    match.group(1)
                    .replace(",", "")
                )

                return value

            except ValueError:

                pass

    return None


# ============================================================
# PREFERENCE EXTRACTION / NORMALIZATION
# ============================================================

def extract_preferences_from_query(query):
    """Extract important product preferences explicitly present in the query."""
    query_lower = query.lower()

    preference_patterns = [
        ("lightweight", [r"\blightweight\b"]),
        ("developer", [r"\bdeveloper\b"]),
        ("programming", [r"\bprogramming\b"]),
        ("software development", [r"\bsoftware development\b"]),
        ("gaming", [r"\bgaming\b"]),
        ("performance", [r"\bperformance\b"]),
        ("student", [r"\bstudent\b"]),
        ("college", [r"\bcollege\b"]),
        ("premium", [r"\bpremium\b"]),
        ("productivity", [r"\bproductivity\b"]),
        ("camera", [r"\bcamera\b"]),
        ("photography", [r"\bphotography\b"]),
        ("battery", [r"\bbattery\b"]),
        ("travel", [r"\btravel\b"]),
        ("blue", [r"\bblue\b"]),
        ("formal", [r"\bformal\b"]),
        ("office", [r"\boffice\b"]),
        ("black", [r"\bblack\b"]),
        ("casual", [r"\bcasual\b"]),
        ("everyday", [r"\beveryday\b"]),
        ("running", [r"\brunning\b"]),
        ("fitness", [r"\bfitness\b"]),
        ("waterproof", [r"\bwaterproof\b"]),
        ("laptop bag", [r"\blaptop\s+bag\b"]),
        ("power bank", [r"\bpower\s+bank\b"]),
        ("USB C", [r"\busb[\s-]?c\b", r"\busbc\b"]),
        ("charging", [r"\bcharging\b"]),
        ("smart home", [r"\bsmart\s+home\b"]),
        ("digital camera", [r"\bdigital\s+camera\b"]),
    ]

    preferences = []
    for preference, patterns in preference_patterns:
        if any(re.search(pattern, query_lower) for pattern in patterns):
            preferences.append(preference)
    return preferences


# ============================================================
# QWEN INTENT EXTRACTION
# ============================================================

def extract_intent(query):

    prompt = f"""
You are an e-commerce search intent extraction system.

Analyze the user's query.

Return ONLY valid JSON.

Required structure:

{{
    "category": "laptops | mobiles | electronics | fashion | shoes | accessories | null",
    "intent": "purchase | information | recommendation | null",
    "max_price": number or null,
    "preferences": []
}}

Rules:

1. Identify the product category.
2. Extract maximum price constraints.
3. Extract important product preferences.
4. Do not invent information.
5. Return ONLY JSON.
6. Do not use markdown.
7. Do not explain your answer.

User query:

{query}
"""

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    try:

        response = requests.post(
            QWEN_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        content = (
            result["choices"][0]["message"]["content"]
            .strip()
        )

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        intent = json.loads(
            content
        )

    except Exception as error:

        print(
            f"Qwen intent extraction error: {error}"
        )

        intent = {}


    intent = normalize_intent(
        intent
    )


    # --------------------------------------------------------
    # Preference normalization
    # --------------------------------------------------------

    detected_preferences = extract_preferences_from_query(query)

    existing_preferences = [
        str(pref).strip()
        for pref in intent.get("preferences", [])
        if str(pref).strip()
    ]

    combined_preferences = []

    for preference in existing_preferences:
        if preference not in combined_preferences:
            combined_preferences.append(preference)

    for preference in detected_preferences:
        if preference not in combined_preferences:
            combined_preferences.append(preference)

    intent["preferences"] = combined_preferences


    # --------------------------------------------------------
    # Category fallback
    # --------------------------------------------------------

    if not intent["category"]:

        detected_category = (
            extract_category_from_query(
                query
            )
        )

        if detected_category:

            intent["category"] = (
                detected_category
            )


    # --------------------------------------------------------
    # Price fallback
    # --------------------------------------------------------

    detected_price = (
        extract_max_price_from_query(
            query
        )
    )

    if detected_price is not None:

        intent["max_price"] = (
            detected_price
        )


    print(
        "Final extracted intent:",
        json.dumps(
            intent,
            indent=2
        )
    )

    return intent

def retrieve_product_context(query, top_k=3):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name="products",
        query=query_embedding,
        limit=top_k
    ).points

    contexts = []

    for result in results:
        product = result.payload

        contexts.append({
            "product_id": str(product["product_id"]),
            "product_name": str(product["product_name"]),
            "category": str(product["category"]),
            "description": str(product["description"]),
            "features": str(product["features"]),
            "use_cases": str(product["use_cases"]),
            "price": float(product["price"]),
            "rating": float(product["rating"]),
            "specifications": str(
                product["specifications"]
            )
        })

    return contexts
# ============================================================
# QWEN-DRIVEN SEMANTIC SEARCH
# ============================================================

def qwen_semantic_search(
    query,
    top_k=10
):

    intent = extract_intent(
        query
    )

    conditions = []


    # Category

    if intent["category"]:

        conditions.append(

            FieldCondition(
                key="category",
                match=MatchValue(
                    value=intent["category"]
                )
            )

        )


    # Price

    if intent["max_price"] is not None:

        conditions.append(

            FieldCondition(
                key="price",
                range=Range(
                    lte=intent["max_price"]
                )
            )

        )


    # Availability

    conditions.append(

        FieldCondition(
            key="availability",
            match=MatchValue(
                value="in_stock"
            )
        )

    )


    search_filter = Filter(
        must=conditions
    )


    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        query_filter=search_filter,

        limit=top_k

    ).points


    print(
        f"Qdrant retrieved {len(results)} candidates."
    )


    return (
        intent,
        results
    )


# ============================================================
# QWEN RE-RANKING
# ============================================================

def rerank_with_qwen(
    query,
    intent,
    results,
    top_k=5
):
    """
    Hybrid reranking.

    Qdrant provides the primary semantic ranking.
    Qwen provides an intent-aware relevance signal.

    Final score:
        70% semantic ranking
        30% Qwen relevance
    """

    if not results:
        return []

    candidate_count = len(results)

    print(
        f"Qwen reranking {candidate_count} candidates"
    )

    # --------------------------------------------------------
    # Build compact candidate representation
    # --------------------------------------------------------

    candidates = []

    for index, result in enumerate(results):

        product = result.payload

        candidates.append({
            "product_id": str(
                product["product_id"]
            ),
            "product_name": str(
                product["product_name"]
            ),
            "category": str(
                product["category"]
            ),
            "price": float(
                product["price"]
            ),
            "features": str(
                product.get(
                    "features",
                    ""
                )
            )[:500],
            "use_cases": str(
                product.get(
                    "use_cases",
                    ""
                )
            )[:300]
        })

    # --------------------------------------------------------
    # Qwen prompt
    # --------------------------------------------------------

    prompt = f"""
You are an e-commerce product relevance evaluator.

USER QUERY:
{query}

USER INTENT:
{json.dumps(intent, ensure_ascii=False)}

CANDIDATE PRODUCTS:
{json.dumps(candidates, ensure_ascii=False)}

Score every candidate from 0 to 100.

Scoring priorities:

1. Match the requested product category.
2. Respect the maximum price.
3. Match the user's intended use case.
4. Match explicit preferences.
5. Match relevant product features.

Do not use rating as the primary factor.

IMPORTANT:

- Use ONLY the provided candidates.
- Never invent product IDs.
- Return EVERY candidate exactly once.
- Use the exact product_id values.
- Return ONLY JSON.
- No explanations.
- Scores must be integers from 0 to 100.

Required format:

{{
  "products": [
    {{
      "product_id": "ID",
      "score": 95
    }}
  ]
}}
"""

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 256
    }

    try:

        response = requests.post(
            QWEN_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        response_data = response.json()

        content = (
            response_data["choices"][0]["message"]["content"]
            .strip()
        )

        # ----------------------------------------------------
        # Clean markdown fences
        # ----------------------------------------------------

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        ranking = json.loads(content)

        ranked_products = ranking.get(
            "products",
            []
        )

        if not isinstance(
            ranked_products,
            list
        ):
            raise ValueError(
                "Invalid Qwen ranking format"
            )

        # ----------------------------------------------------
        # Create lookup of original Qdrant results
        # ----------------------------------------------------

        result_lookup = {
            str(
                result.payload["product_id"]
            ): result
            for result in results
        }

        # ----------------------------------------------------
        # Extract Qwen scores
        # ----------------------------------------------------

        qwen_scores = {}

        for item in ranked_products:

            if not isinstance(
                item,
                dict
            ):
                continue

            product_id = str(
                item.get(
                    "product_id",
                    ""
                )
            )

            if product_id not in result_lookup:
                continue

            try:

                score = float(
                    item.get(
                        "score",
                        0
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            score = max(
                0,
                min(
                    100,
                    score
                )
            )

            qwen_scores[product_id] = (
                score / 100.0
            )

        # ----------------------------------------------------
        # Validate Qwen output
        # ----------------------------------------------------

        if not qwen_scores:

            raise ValueError(
                "Qwen returned no valid product scores"
            )

        # ----------------------------------------------------
        # Hybrid ranking
        #
        # Semantic rank is intentionally dominant.
        # This prevents Qwen from destroying a strong
        # semantic result because of a bad generation.
        # ----------------------------------------------------

        hybrid_results = []

        denominator = max(
            candidate_count - 1,
            1
        )

        for index, result in enumerate(results):

            product_id = str(
                result.payload["product_id"]
            )

            # Semantic rank score:
            #
            # first result = 1.0
            # last result  = 0.0
            #
            # This keeps the original semantic ranking
            # stable while allowing Qwen to adjust it.

            semantic_score = (
                1.0
                - (
                    index
                    / denominator
                )
            )

            qwen_score = qwen_scores.get(
                product_id,
                0.0
            )

            hybrid_score = (
                0.70 * semantic_score
                +
                0.30 * qwen_score
            )

            hybrid_results.append(
                (
                    hybrid_score,
                    result
                )
            )

        # ----------------------------------------------------
        # Sort by hybrid score
        # ----------------------------------------------------

        hybrid_results.sort(
            key=lambda item: item[0],
            reverse=True
        )

        ranked_results = [
            result
            for score, result
            in hybrid_results
        ]

        # ----------------------------------------------------
        # Final deterministic safety filtering
        # ----------------------------------------------------

        safe_results = []

        for result in ranked_results:

            product = result.payload

            # Category
            if intent.get("category"):

                if (
                    str(
                        product["category"]
                    ).lower()
                    != str(
                        intent["category"]
                    ).lower()
                ):
                    continue

            # Budget
            if (
                intent.get("max_price")
                is not None
            ):

                if (
                    float(
                        product["price"]
                    )
                    >
                    float(
                        intent["max_price"]
                    )
                ):
                    continue

            # Availability
            if (
                str(
                    product.get(
                        "availability",
                        ""
                    )
                ).lower()
                != "in_stock"
            ):
                continue

            safe_results.append(
                result
            )

        print(
            f"Hybrid reranking complete: "
            f"{len(safe_results)} safe candidates"
        )

        return safe_results[:top_k]

    except Exception as error:

        print(
            f"Qwen reranking failed: {error}"
        )

        # ----------------------------------------------------
        # Safe fallback:
        # original Qdrant semantic ranking
        # ----------------------------------------------------

        safe_results = []

        for result in results:

            product = result.payload

            if intent.get("category"):

                if (
                    str(
                        product["category"]
                    ).lower()
                    != str(
                        intent["category"]
                    ).lower()
                ):
                    continue

            if (
                intent.get("max_price")
                is not None
            ):

                if (
                    float(
                        product["price"]
                    )
                    >
                    float(
                        intent["max_price"]
                    )
                ):
                    continue

            if (
                str(
                    product.get(
                        "availability",
                        ""
                    )
                ).lower()
                != "in_stock"
            ):
                continue

            safe_results.append(
                result
            )

        return safe_results[:top_k]


# ============================================================
# PRODUCT SERIALIZATION
# ============================================================

def serialize_product(product):

    if product is None:
        return None


    result = {}

    for key, value in product.items():

        if pd.isna(value):
            result[key] = None

        else:
            result[key] = value


    # Convert numeric fields

    if result.get("price") is not None:

        result["price"] = float(
            result["price"]
        )


    if result.get("rating") is not None:

        result["rating"] = float(
            result["rating"]
        )


    if result.get("review_count") is not None:

        result["review_count"] = int(
            result["review_count"]
        )


    return result


# ============================================================
# PRODUCT LOOKUP
# ============================================================

def get_product_by_id(product_id):

    product = df[
        df["product_id"].astype(str)
        == str(product_id)
    ]


    if product.empty:

        return None


    return serialize_product(
        product.iloc[0].to_dict()
    )


# ============================================================
# CROSS-SELL RECOMMENDATIONS
# ============================================================

def get_cross_sell_recommendations(
    product_id
):

    product = df[
        df["product_id"].astype(str)
        == str(product_id)
    ]


    if product.empty:

        return []


    related_ids = product.iloc[0].get(
        "related_product_ids",
        ""
    )


    if pd.isna(related_ids) or not related_ids:

        return []


    related_ids = str(
        related_ids
    ).split(";")


    recommendations = df[
        df["product_id"].astype(str).isin(
            related_ids
        )
    ]


    return [

        serialize_product(
            row.to_dict()
        )

        for _, row in recommendations.iterrows()

    ]


# ============================================================
# UPSELL RECOMMENDATIONS
# ============================================================

def get_upsell_recommendations(
    product_id
):

    product = df[
        df["product_id"].astype(str)
        == str(product_id)
    ]


    if product.empty:

        return []


    upsell_ids = product.iloc[0].get(
        "upsell_product_ids",
        ""
    )


    if pd.isna(upsell_ids) or not upsell_ids:

        return []


    upsell_ids = str(
        upsell_ids
    ).split(";")


    recommendations = df[
        df["product_id"].astype(str).isin(
            upsell_ids
        )
    ]


    return [

        serialize_product(
            row.to_dict()
        )

        for _, row in recommendations.iterrows()

    ]


# ============================================================
# COMPLETE RECOMMENDATIONS
# ============================================================

def get_recommendations(
    product_id
):

    product = get_product_by_id(
        product_id
    )


    if product is None:

        return None


    cross_sell = (
        get_cross_sell_recommendations(
            product_id
        )
    )


    upsell = (
        get_upsell_recommendations(
            product_id
        )
    )


    return {

        "product": product,

        "cross_sell": cross_sell,

        "upsell": upsell

    }


# ============================================================
# QWEN UPSELL EXPLANATION
# ============================================================

def explain_upsell_with_qwen(
    base_product,
    upsell_product
):

    if not base_product or not upsell_product:

        return (
            "There is not enough product information "
            "to explain this recommendation."
        )


    base_context = {

        "product_name": str(
            base_product.get(
                "product_name",
                ""
            )
        ),

        "price": float(
            base_product.get(
                "price",
                0
            )
        ),

        "rating": float(
            base_product.get(
                "rating",
                0
            )
        ),

        "features": str(
            base_product.get(
                "features",
                ""
            )
        ),

        "description": str(
            base_product.get(
                "description",
                ""
            )
        )

    }


    upsell_context = {

        "product_name": str(
            upsell_product.get(
                "product_name",
                ""
            )
        ),

        "price": float(
            upsell_product.get(
                "price",
                0
            )
        ),

        "rating": float(
            upsell_product.get(
                "rating",
                0
            )
        ),

        "features": str(
            upsell_product.get(
                "features",
                ""
            )
        ),

        "description": str(
            upsell_product.get(
                "description",
                ""
            )
        )

    }


    prompt = f"""
You are an e-commerce recommendation assistant.

Compare these two products.

Selected product:

{json.dumps(base_context, indent=2)}

Recommended upgrade:

{json.dumps(upsell_context, indent=2)}

Explain in 1-2 concise sentences why the recommended
product could provide additional value.

IMPORTANT:

1. Use ONLY information provided above.
2. Do not invent specifications.
3. Do not invent features.
4. Do not invent prices.
5. Do not invent benefits.
6. Mention trade-offs if relevant.
7. If there is not enough information to identify
   an advantage, say so clearly.
"""


    payload = {

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "temperature": 0

    }


    try:

        response = requests.post(

            QWEN_URL,

            json=payload,

            timeout=60

        )


        response.raise_for_status()


        result = response.json()


        explanation = (
            result["choices"][0]["message"]["content"]
            .strip()
        )


        return explanation


    except Exception as error:

        print(
            f"Qwen recommendation explanation error: {error}"
        )


        return (
            "Additional product information is available, "
            "but an AI explanation could not be generated."
        )


# ============================================================
# UNIFIED SEARCH
# ============================================================

def search_products(
    query,
    top_k=5
):

    # Retrieve a small candidate pool for efficient
    # intent-aware reranking.
    candidate_count = max(
        top_k + 1,
        6
    )


    # --------------------------------------------------------
    # PASS 1
    # Intent extraction + semantic retrieval
    # --------------------------------------------------------

    intent, candidates = (
        qwen_semantic_search(

            query,

            candidate_count

        )
    )


    # --------------------------------------------------------
    # PASS 2
    # Qwen re-ranking
    # --------------------------------------------------------

    final_results = rerank_with_qwen(

        query,

        intent,

        candidates,

        top_k=top_k

    )


    return {

        "query": query,

        "intent": intent,

        "results": final_results

    }   