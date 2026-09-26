import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)



df = pd.read_csv("products.csv")

print("Dataset loaded successfully!")
print("Number of products:", len(df))


def create_embedding_text(row):
    return (
        f"Product name: {row['product_name']}. "
        f"Category: {row['category']}. "
        f"Description: {row['description']} "
        f"Features: {row['features']}. "
        f"Tags: {row['tags']}. "
        f"Use cases: {row['use_cases']}."
    )


df["embedding_text"] = df.apply(create_embedding_text, axis=1)


print("\nLoading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Embedding model loaded successfully!")


print("\nGenerating embeddings...")

embeddings = model.encode(
    df["embedding_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbeddings generated successfully!")
print("Number of embeddings:", len(embeddings))
print("Dimensions:", embeddings.shape[1])


print("\nConnecting to Qdrant...")

client = QdrantClient(path="./qdrant_data")

print("Connected to Qdrant!")


collection_name = "products"

# Delete the collection if it already exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print(f"Collection '{collection_name}' created!")


points = []

for index, row in df.iterrows():

    point = PointStruct(
        id=index + 1,
        vector=embeddings[index].tolist(),

        payload={
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "brand": row["brand"],
            "description": row["description"],
            "features": row["features"],
            "tags": row["tags"],
            "use_cases": row["use_cases"],
            "price": float(row["price"]),
            "currency": row["currency"],
            "rating": float(row["rating"]),
            "review_count": int(row["review_count"]),
            "availability": row["availability"],
            "specifications": row["specifications"],
            "related_product_ids": row["related_product_ids"],
            "upsell_product_ids": row["upsell_product_ids"],
            "image_url": row["image_url"]
        }
    )

    points.append(point)


print("\nUploading products to Qdrant...")

client.upsert(
    collection_name=collection_name,
    points=points
)

print("Products uploaded successfully!")


collection_info = client.get_collection(collection_name)

print("\nQdrant collection information:")
print("Vectors stored:", collection_info.points_count)

#quadrant verification 

# --------------------------------------------------
# 10. Semantic search with metadata filters
# --------------------------------------------------

query = "lightweight laptop for software development"

print(f"\nSearch query: {query}")

# Convert the query into an embedding
query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()


# Create filters
search_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="laptops")
        ),
        FieldCondition(
            key="price",
            range=Range(lte=50000)
        ),
        FieldCondition(
            key="availability",
            match=MatchValue(value="in_stock")
        )
    ]
)

# Search Qdrant
search_results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    query_filter=search_filter,
    limit=5
).points


# Display results
print("\nTop 5 filtered semantic search results:\n")

for rank, result in enumerate(search_results, start=1):

    product = result.payload

    print(f"{rank}. {product['product_name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Availability: {product['availability']}")
    print(f"   Similarity score: {result.score:.4f}")
    print()

    # --------------------------------------------------
# 11. Traditional keyword search
# --------------------------------------------------

def keyword_search(query, dataframe, top_k=5):
    """
    Simple keyword-based product search with
    structured filtering.
    """

    # Convert query into individual words
    query_terms = re.findall(r"\b\w+\b", query.lower())

    results = []

    for _, row in dataframe.iterrows():

        # Apply the same filters used by Qdrant
        if row["category"].lower() != "laptops":
            continue

        if row["price"] > 50000:
            continue

        if row["availability"].lower() != "in_stock":
            continue

        # Searchable product text
        searchable_text = (
            f"{row['product_name']} "
            f"{row['category']} "
            f"{row['subcategory']} "
            f"{row['description']} "
            f"{row['features']} "
            f"{row['tags']} "
            f"{row['use_cases']}"
        ).lower()

        # Count matching query terms
        score = sum(
            1 for term in query_terms
            if term in searchable_text
        )

        if score > 0:
            results.append((score, row))

    # Highest keyword score first
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:top_k]

# --------------------------------------------------
# 12. Test keyword search
# --------------------------------------------------

keyword_query = "lightweight laptop for software development"

keyword_results = keyword_search(
    keyword_query,
    df,
    top_k=5
)

print("\nTraditional keyword search results:\n")

for rank, (score, product) in enumerate(keyword_results, start=1):

    print(f"{rank}. {product['product_name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Keyword score: {score}")
    print()

# --------------------------------------------------
# 13. Normalize Qwen intent
# --------------------------------------------------

def normalize_intent(intent):
    """
    Normalize Qwen's output so it matches our dataset.
    """

    # Normalize category
    category = intent.get("category")

    if category:
        category = category.lower().strip()

        category_map = {
            "laptop": "laptops",
            "mobile": "mobiles",
            "phone": "mobiles",
            "smartphone": "mobiles",
            "shoe": "shoes",
            "accessory": "accessories",
            "fashion": "fashion",
            "electronic": "electronics"
        }

        category = category_map.get(category, category)

    # Convert price to a number
    max_price = intent.get("max_price")

    if max_price is not None:
        try:
            max_price = float(max_price)
        except (ValueError, TypeError):
            max_price = None

    # Normalize preferences
    preferences = intent.get("preferences", [])

    if isinstance(preferences, str):
        preferences = [preferences]

    return {
        "category": category,
        "intent": intent.get("intent"),
        "max_price": max_price,
        "preferences": preferences
    }


# --------------------------------------------------
# 14. Test intent normalization
# --------------------------------------------------

qwen_output = """
{
    "category": "laptop",
    "intent": "purchase",
    "max_price": "50000",
    "preferences": [
        "lightweight",
        "software development"
    ]
}
"""

intent = json.loads(qwen_output)

normalized_intent = normalize_intent(intent)

print("\nNormalized intent:")
print(json.dumps(normalized_intent, indent=2))


import urllib.request
import json

url = "http://127.0.0.1:8080/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Explain semantic search in one simple sentence."
        }
    ],
    "temperature": 0
}

data = json.dumps(payload).encode("utf-8")

request = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))

print(result["choices"][0]["message"]["content"])


def extract_intent(query):
    """
    Send a user query to Qwen and extract structured search intent.
    """

    url = "http://127.0.0.1:8080/v1/chat/completions"

    prompt = f"""
You are an e-commerce search intent extraction system.

Your task is to convert the customer's natural-language shopping query
into a precise structured intent.

Return ONLY valid JSON.
Do not use markdown.
Do not explain your answer.

Required JSON structure:

{{
    "category": "laptops | mobiles | electronics | fashion | shoes | accessories | null",
    "intent": "purchase | information | recommendation | null",
    "max_price": number or null,
    "preferences": []
}}

CATEGORY RULES:

- laptop, laptops, notebook, developer laptop, gaming laptop
  -> laptops

- phone, mobile, smartphone, smartphones
  -> mobiles

- electronics, electronic devices, camera, power bank, cable,
  charger, smart home device
  -> electronics

- shirt, clothing, fashion, jeans, jacket, formal wear
  -> fashion

- shoes, sneakers, footwear, running shoes
  -> shoes

- bag, backpack, watch, accessories, mouse, keyboard
  -> accessories

IMPORTANT:
Choose the category based on the actual product being requested,
not merely a word describing its intended use.

For example:
"waterproof laptop bag for office"
-> accessories
NOT laptops.

"USB C cable for charging"
-> electronics.

"camera for photography"
-> electronics.

PREFERENCE RULES:

Extract meaningful product requirements such as:

- color
- size
- weight/lightweight
- performance
- gaming
- programming
- software development
- photography
- camera
- battery
- waterproof
- formal
- casual
- office
- college/student
- portability
- productivity
- smart home
- charging
- USB-C
- etc.

Preserve important product nouns when they are useful.
For example:
"USB C cable for charging"
should contain a preference related to USB-C/cable/charging.

PRICE RULES:

If the query contains an explicit maximum price such as:

"under 50000"
"below ₹50,000"
"within 50k"
"less than 80000"

extract the numeric maximum price.

If there is no explicit maximum price, return null.

INTENT RULES:

- "buy", "looking for", "need", "want", "find me"
  -> purchase

- "what", "does it support", "is it compatible", "how"
  -> information

- "recommend", "suggest", "which should I buy"
  -> recommendation

Do not invent preferences that are not present or clearly implied
by the query.

Examples:

Query:
"lightweight laptop for software development under 50000"

Output:
{{
    "category": "laptops",
    "intent": "purchase",
    "max_price": 50000,
    "preferences": ["lightweight", "software development"]
}}

Query:
"blue formal shirt for office"

Output:
{{
    "category": "fashion",
    "intent": "purchase",
    "max_price": null,
    "preferences": ["blue", "formal", "office"]
}}

Query:
"waterproof laptop bag for office"

Output:
{{
    "category": "accessories",
    "intent": "purchase",
    "max_price": null,
    "preferences": ["waterproof", "laptop bag", "office"]
}}

Query:
"USB C cable for charging under 1500"

Output:
{{
    "category": "electronics",
    "intent": "purchase",
    "max_price": 1500,
    "preferences": ["USB C", "cable", "charging"]
}}

Now analyze this query:

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

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    qwen_text = result["choices"][0]["message"]["content"]

    # Remove possible markdown code fences
    qwen_text = qwen_text.replace("```json", "").replace("```", "").strip()

    intent = json.loads(qwen_text)

    return normalize_intent(intent)

def qwen_semantic_search(query, top_k=5):
    """
    Use Qwen to extract intent, then use that intent
    to filter and search products in Qdrant.
    """

    # 1. Extract intent using Qwen
    intent = extract_intent(query)

    # 2. Build Qdrant filters dynamically
    conditions = []

    if intent["category"]:
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=intent["category"])
            )
        )

    if intent["max_price"] is not None:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(lte=intent["max_price"])
            )
        )

    # Only return products currently available
    conditions.append(
        FieldCondition(
            key="availability",
            match=MatchValue(value="in_stock")
        )
    )

    search_filter = Filter(must=conditions)

    # 3. Convert the original query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # 4. Search Qdrant using the embedding + Qwen filters
    results = client.query_points(
        collection_name="products",
        query=query_embedding,
        query_filter=search_filter,
        limit=top_k
    ).points

    return intent, results


def rerank_with_qwen(
    query,
    intent,
    results,
    top_k=5
):
    """
    Rerank semantic-search candidates using Qwen.

    Qwen receives compact product information and returns
    a relevance score for each candidate.
    """

    if not results:
        return []

    print(f"Qwen reranking {len(results)} candidates")

    # ---------------------------------------------------------
    # Build compact candidate information
    # ---------------------------------------------------------

    candidates = []

    for result in results:
        product = result.payload

        candidates.append({
            "product_id": str(product["product_id"]),
            "product_name": str(product["product_name"]),
            "category": str(product["category"]),
            "price": float(product["price"]),
            "rating": float(product["rating"]),
            "features": str(product.get("features", "")),
            "use_cases": str(product.get("use_cases", ""))
        })

    # ---------------------------------------------------------
    # Qwen ranking prompt
    # ---------------------------------------------------------

    prompt = f"""
You are an e-commerce product relevance reranker.

USER QUERY:
{query}

DETECTED INTENT:
{json.dumps(intent, ensure_ascii=False)}

CANDIDATE PRODUCTS:
{json.dumps(candidates, ensure_ascii=False)}

Your task is to score EVERY candidate according to how well
it satisfies the user's query.

RANKING PRIORITY:

1. Respect the requested product category.
2. Respect the maximum budget if one is provided.
3. Match the user's intended use case.
4. Match explicit user preferences.
5. Match relevant product features.
6. Use rating only as a secondary signal.

IMPORTANT RULES:

- Only use products from the candidate list.
- Never invent a product.
- Never invent a product ID.
- Return every candidate exactly once.
- Use the exact product_id provided.
- A product that violates the user's explicit category or
  maximum budget should receive a very low score.
- Do not favor a product simply because it has a higher rating.
- Relevance to the user's query is more important than rating.
- Return ONLY valid JSON.
- Do not include explanations.

Return exactly this format:

{{
    "ranked_products": [
        {{
            "product_id": "ID1",
            "score": 95
        }},
        {{
            "product_id": "ID2",
            "score": 87
        }}
    ]
}}

Scores must be integers from 0 to 100.
Higher score means more relevant.
"""

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 512
    }

    try:

        # -----------------------------------------------------
        # Call Qwen
        # -----------------------------------------------------

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
        print("RAW QWEN RESPONSE:")
        print(content)
        print("END RAW QWEN RESPONSE")

        # -----------------------------------------------------
        # Remove markdown code fences if Qwen adds them
        # -----------------------------------------------------

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        ranking = json.loads(content)

        ranked_products = ranking.get(
            "ranked_products",
            []
        )

        # -----------------------------------------------------
        # Validate Qwen output
        # -----------------------------------------------------

        result_lookup = {
            str(result.payload["product_id"]): result
            for result in results
        }

        scored_results = []
        used_ids = set()

        for item in ranked_products:

            if not isinstance(item, dict):
                continue

            product_id = str(
                item.get("product_id", "")
            )

            if product_id not in result_lookup:
                continue

            if product_id in used_ids:
                continue

            try:
                score = float(
                    item.get("score", 0)
                )
            except (TypeError, ValueError):
                score = 0

            scored_results.append(
                (
                    score,
                    result_lookup[product_id]
                )
            )

            used_ids.add(product_id)

        # -----------------------------------------------------
        # Sort by Qwen relevance score
        # -----------------------------------------------------

        scored_results.sort(
            key=lambda item: item[0],
            reverse=True
        )
        print("Qwen scores:")
        for score, result in scored_results:
            print(
            f"  {result.payload['product_id']} -> {score}"
    )

        final_results = [
            result
            for score, result in scored_results
        ]

        # -----------------------------------------------------
        # Safety: append candidates Qwen accidentally omitted
        # -----------------------------------------------------

        for result in results:

            product_id = str(
                result.payload["product_id"]
            )

            if product_id not in used_ids:

                final_results.append(result)

                used_ids.add(product_id)

        # -----------------------------------------------------
        # Final deterministic safety filtering
        # -----------------------------------------------------

        safe_results = []

        for result in final_results:

            product = result.payload

            # Category filter
            if intent.get("category"):

                if (
                    str(product["category"]).lower()
                    != str(intent["category"]).lower()
                ):
                    continue

            # Budget filter
            if intent.get("max_price") is not None:

                if (
                    float(product["price"])
                    > float(intent["max_price"])
                ):
                    continue

            # Availability filter
            if (
                str(product.get("availability", "")).lower()
                != "in_stock"
            ):
                continue

            safe_results.append(result)

        print(
            f"Qwen reranking complete: "
            f"{len(safe_results)} safe candidates"
        )

        return safe_results[:top_k]

    except Exception as error:

        print(
            f"Qwen reranking error: {error}"
        )

        # -----------------------------------------------------
        # Safe fallback:
        # return original semantic ranking
        # -----------------------------------------------------

        safe_results = []

        for result in results:

            product = result.payload

            if intent.get("category"):

                if (
                    str(product["category"]).lower()
                    != str(intent["category"]).lower()
                ):
                    continue

            if intent.get("max_price") is not None:

                if (
                    float(product["price"])
                    > float(intent["max_price"])
                ):
                    continue

            if (
                str(product.get("availability", "")).lower()
                != "in_stock"
            ):
                continue

            safe_results.append(result)

        return safe_results[:top_k]

def search_products(query, top_k=5):
    """
    Complete two-pass AI product search.

    Pass 1:
        Qwen intent extraction
        → Qdrant semantic retrieval

    Pass 2:
        Qwen re-ranking
        → final products
    """

    # Pass 1: retrieve candidates
    intent, candidates = qwen_semantic_search(
        query,
        top_k=10
    )

    # Pass 2: re-rank candidates
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


query = "I need a lightweight laptop for software development under ₹50,000"

# Retrieve 10 candidates using Qwen intent + semantic search
intent, candidates = qwen_semantic_search(query, top_k=10)

print("\nCandidates retrieved from Qdrant:")
for i, result in enumerate(candidates, 1):
    print(
        f"{i}. {result.payload['product_name']} "
        f"(score: {result.score:.4f})"
    )

# Re-rank those candidates using Qwen
reranked_results = rerank_with_qwen(
    query,
    intent,
    candidates,
    top_k=5
)

print("\nQwen Re-ranked Results:")

for i, result in enumerate(reranked_results, 1):
    product = result.payload

    print(f"\n{i}. {product['product_name']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Rating: {product['rating']}")
    print(f"   Original similarity: {result.score:.4f}")


search_result = search_products(
    "I need a lightweight laptop for software development under ₹50,000"
)

print("\n=== FINAL SEARCH RESULTS ===")

print("\nIntent:")
print(json.dumps(search_result["intent"], indent=2))

print("\nProducts:")

for i, result in enumerate(search_result["results"], 1):
    product = result.payload

    print(f"\n{i}. {product['product_name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Rating: {product['rating']}")
    print(f"   Similarity: {result.score:.4f}")

# Inspect product relationships

product = df.iloc[0]

print("\n=== PRODUCT RELATIONSHIP TEST ===")

print("Product:")
print(product["product_name"])

print("\nRelated product IDs:")
print(product["related_product_ids"])

print("\nUpsell product IDs:")
print(product["upsell_product_ids"])

def get_cross_sell_recommendations(product_id):
    """
    Get complementary products related to the selected product.
    """

    product = df[df["product_id"] == product_id]

    if product.empty:
        return []

    related_ids = product.iloc[0]["related_product_ids"]

    if not related_ids:
        return []

    related_ids = related_ids.split(";")

    recommendations = df[
        df["product_id"].isin(related_ids)
    ]

    return recommendations

recommendations = get_cross_sell_recommendations("MOB001")

print("\n=== CROSS-SELL RECOMMENDATIONS ===")

for _, product in recommendations.iterrows():
    print(f"\n{product['product_name']}")
    print(f"Category: {product['category']}")
    print(f"Price: ₹{product['price']}")

def get_upsell_recommendations(product_id):
    """
    Get higher-value alternative products for the selected product.
    """

    product = df[df["product_id"] == product_id]

    if product.empty:
        return []

    upsell_ids = product.iloc[0]["upsell_product_ids"]

    if not upsell_ids:
        return []

    upsell_ids = upsell_ids.split(";")

    recommendations = df[
        df["product_id"].isin(upsell_ids)
    ]

    return recommendations

upsells = get_upsell_recommendations("MOB001")

print("\n=== UPSELL RECOMMENDATIONS ===")

for _, product in upsells.iterrows():
    print(f"\n{product['product_name']}")
    print(f"Category: {product['category']}")
    print(f"Price: ₹{product['price']}")
    print(f"Rating: {product['rating']}")

def get_recommendations(product_id):
    """
    Get both cross-sell and upsell recommendations
    for a selected product.
    """

    cross_sell = get_cross_sell_recommendations(product_id)
    upsell = get_upsell_recommendations(product_id)

    return {
        "cross_sell": cross_sell,
        "upsell": upsell
    }

def explain_upsell_with_qwen(base_product, upsell_product):
    """
    Generate a grounded explanation for an upsell recommendation.
    """

    prompt = f"""
You are an e-commerce recommendation assistant.

Compare these two products.

Selected product:
{json.dumps(base_product, indent=2)}

Recommended upgrade:
{json.dumps(upsell_product, indent=2)}

Explain in 1-2 sentences why the recommended product
could provide additional value.

IMPORTANT:
- Use ONLY information provided above.
- Do not invent specifications, features, prices, or benefits.
- If there is not enough information to identify an advantage,
  say so clearly.
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

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()

base = df[df["product_id"] == "MOB001"].iloc[0]
upsell = df[df["product_id"] == "MOB007"].iloc[0]

base_product = {
    "product_name": str(base["product_name"]),
    "price": float(base["price"]),
    "rating": float(base["rating"]),
    "features": str(base["features"]),
    "description": str(base["description"])
}

upsell_product = {
    "product_name": str(upsell["product_name"]),
    "price": float(upsell["price"]),
    "rating": float(upsell["rating"]),
    "features": str(upsell["features"]),
    "description": str(upsell["description"])
}

explanation = explain_upsell_with_qwen(
    base_product,
    upsell_product
)

print("\n=== QWEN UPSELL EXPLANATION ===")
print(explanation)


recommendations = get_recommendations("MOB001")

print("\n=== UNIFIED RECOMMENDATIONS ===")

print("\nCross-sell:")

for _, product in recommendations["cross_sell"].iterrows():
    print(
        f"- {product['product_name']} "
        f"(₹{product['price']})"
    )

print("\nUpsell:")

for _, product in recommendations["upsell"].iterrows():
    print(
        f"- {product['product_name']} "
        f"(₹{product['price']}, "
        f"Rating: {product['rating']})"
    )

def retrieve_product_context(query, top_k=3):
    """
    Retrieve relevant product information from Qdrant
    for grounded generation.
    """

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
            "specifications": str(product["specifications"])
        })

    return contexts

# Task 13 - Inspect retrieved product context

sample_product = df[df["product_id"] == "MOB001"].iloc[0]

product_context = {
    "product_id": str(sample_product["product_id"]),
    "product_name": str(sample_product["product_name"]),
    "category": str(sample_product["category"]),
    "description": str(sample_product["description"]),
    "features": str(sample_product["features"]),
    "use_cases": str(sample_product["use_cases"]),
    "price": float(sample_product["price"]),
    "rating": float(sample_product["rating"]),
    "specifications": str(sample_product["specifications"])
}

print("\n=== RAG PRODUCT CONTEXT ===")
print(json.dumps(product_context, indent=2))

query = "What are the features of a good smartphone for daily student use?"

contexts = retrieve_product_context(query, top_k=3)

print("\n=== RETRIEVED RAG CONTEXT ===")

for i, context in enumerate(contexts, 1):
    print(f"\nProduct {i}:")
    print(json.dumps(context, indent=2))

def generate_grounded_answer(query, contexts):
    """
    Generate an answer using only retrieved product context.
    """

    context_text = json.dumps(contexts, indent=2)

    prompt = f"""
You are an e-commerce product assistant.

Answer the user's question using ONLY the product information
provided in the context below.

USER QUESTION:
{query}

PRODUCT CONTEXT:
{context_text}

Rules:
- Use only facts present in the product context.
- Do not invent specifications, features, prices, ratings, or products.
- If the context does not contain enough information to answer,
  clearly say that the available product information is insufficient.
- Mention the relevant product name when appropriate.
- Keep the answer concise and useful.
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

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()

query = "Which smartphone has the highest camera resolution among these products?"

contexts = [
    {
        "product_id": "MOB001",
        "product_name": "Novatek Budget Smartphone 001",
        "features": "4GB RAM; 256GB storage; 50MP main camera; 5000mAh battery; 90Hz display; 5G connectivity"
    },
    {
        "product_id": "MOB067",
        "product_name": "Novatek Budget Smartphone 067",
        "features": "4GB RAM; 512GB storage; 108MP main camera; 5000mAh battery; 90Hz display; 5G connectivity"
    },
    {
        "product_id": "MOB085",
        "product_name": "Novatek Budget Smartphone 085",
        "features": "6GB RAM; 256GB storage; 64MP main camera; 5000mAh battery; 60Hz display; 5G connectivity"
    }
]

answer = generate_grounded_answer(
    query,
    contexts
)

print("\n=== GROUNDED QWEN ANSWER ===")
print(answer)

user_profile = {
    "user_id": "USER001",

    "preferred_categories": [
        "laptops",
        "accessories"
    ],

    "preferred_price_range": {
        "min": 20000,
        "max": 60000
    },

    "purchase_history": [
        "LAP012",
        "ACC034"
    ]
}

print("\n=== USER PROFILE ===")
print(json.dumps(user_profile, indent=2))

def personalize_results(results, user_profile):
    """
    Adjust product ranking using basic user preferences.
    """

    personalized_results = []

    min_price = user_profile["preferred_price_range"]["min"]
    max_price = user_profile["preferred_price_range"]["max"]

    preferred_categories = [
        category.lower()
        for category in user_profile["preferred_categories"]
    ]

    purchase_history = set(user_profile["purchase_history"])

    for result in results:
        product = result.payload

        # Start with the semantic similarity score
        score = result.score

        # Category preference boost
        if product["category"].lower() in preferred_categories:
            score += 0.05

        # Preferred price range boost
        if min_price <= product["price"] <= max_price:
            score += 0.03

        # Avoid recommending something already purchased
        if product["product_id"] in purchase_history:
            score -= 0.10

        personalized_results.append(
            (score, result)
        )

    # Highest personalized score first
    personalized_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [result for score, result in personalized_results]

# Create a mixed-category candidate set
mixed_results = []

for product_id in ["LAP012", "ACC034", "ACC040", "MOB001", "LAP030"]:
    product = df[df["product_id"] == product_id].iloc[0]

    # Create a simple result-like object
    class MockResult:
        pass

    result = MockResult()
    result.payload = product.to_dict()
    result.score = 0.70

    mixed_results.append(result)


# Test profile strongly prefers accessories between ₹1,000–₹5,000
test_profile = {
    "user_id": "USER002",
    "preferred_categories": ["accessories"],
    "preferred_price_range": {
        "min": 1000,
        "max": 5000
    },
    "purchase_history": []
}


# Before personalization
print("=== BEFORE PERSONALIZATION ===")

for i, result in enumerate(mixed_results, 1):
    product = result.payload

    print(
        f"{i}. {product['product_name']} | "
        f"{product['category']} | "
        f"₹{product['price']}"
    )


# After personalization
personalized_results = personalize_results(
    mixed_results,
    test_profile
)

print("\n=== AFTER PERSONALIZATION ===")

for i, result in enumerate(personalized_results, 1):
    product = result.payload

    print(
        f"{i}. {product['product_name']} | "
        f"{product['category']} | "
        f"₹{product['price']}"
    )

query = "I need a laptop for software development"

search_result = search_products(query)

personalized_results = personalize_results(
    search_result["results"],
    user_profile
)

print("\n=== PERSONALIZED RESULTS ===")

for i, result in enumerate(personalized_results, 1):
    product = result.payload

    print(f"\n{i}. {product['product_name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Rating: {product['rating']}")


test_profile = {
    "user_id": "USER002",

    "preferred_categories": [
        "accessories"
    ],

    "preferred_price_range": {
        "min": 1000,
        "max": 5000
    },

    "purchase_history": []
}

personalized_results = personalize_results(
    search_result["results"],
    test_profile
)

print("\n=== PERSONALIZATION TEST ===")

for i, result in enumerate(personalized_results, 1):
    product = result.payload

    print(f"\n{i}. {product['product_name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price: ₹{product['price']}")
    print(f"   Rating: {product['rating']}")

# Use ONLY this product as the retrieved context
test_context = [
    {
        "product_id": "MOB001",
        "product_name": "Novatek Budget Smartphone 001",
        "category": "mobiles",
        "description": "Novatek Budget Smartphone 001 is designed for daily use, student, balancing performance, display quality, camera capability and battery life.",
        "features": "4GB RAM; 256GB storage; 50MP main camera; 5000mAh battery; 90Hz display; 5G connectivity",
        "use_cases": "daily use; student",
        "price": 9400.0,
        "rating": 4.5,
        "specifications": "{\"ram_gb\": 4, \"storage_gb\": 256, \"battery_mah\": 5000, \"camera_mp\": 50, \"screen_refresh_hz\": 90, \"network\": \"5G\"}"
    }
]

# Ask for information that DOES NOT exist in the context
test_query = "Does this smartphone have NFC support?"

answer = generate_grounded_answer(
    test_query,
    test_context
)

print("=== HALLUCINATION TEST ===")
print("Question:", test_query)
print("Answer:", answer)
