from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.search_service import (
    search_products,
    extract_intent,
    get_recommendations,
    get_cross_sell_recommendations,
    get_upsell_recommendations,
    explain_upsell_with_qwen,
    model,
    df,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="VANTOS AI E-Commerce API",
    description=(
        "AI-powered E-Commerce Product Discovery "
        "and Recommendation Engine"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class SearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language product query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of products to return",
    )


class IntentRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
    )


class RecommendationRequest(BaseModel):

    product_id: str = Field(
        ...,
        min_length=1,
    )


class UpsellRequest(BaseModel):

    product_id: str = Field(
        ...,
        min_length=1,
    )


class CrossSellRequest(BaseModel):

    product_id: str = Field(
        ...,
        min_length=1,
    )


class EmbeddingRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )


# ============================================================
# HELPERS
# ============================================================

def serialize_product(product):

    return {
        "product_id": str(
            product["product_id"]
        ),

        "product_name": str(
            product["product_name"]
        ),

        "category": str(
            product["category"]
        ),

        "subcategory": str(
            product.get("subcategory", "")
        ),

        "brand": str(
            product.get("brand", "")
        ),

        "description": str(
            product.get("description", "")
        ),

        "features": str(
            product.get("features", "")
        ),

        "tags": str(
            product.get("tags", "")
        ),

        "use_cases": str(
            product.get("use_cases", "")
        ),

        "price": float(
            product["price"]
        ),

        "currency": str(
            product.get("currency", "INR")
        ),

        "rating": float(
            product["rating"]
        ),

        "review_count": int(
            product.get("review_count", 0)
        ),

        "availability": str(
            product.get("availability", "")
        ),

        "specifications": str(
            product.get("specifications", "")
        ),

        "image_url": str(
            product.get("image_url", "")
        ),
    }


def serialize_dataframe_products(products):

    return [
        serialize_product(
            row
        )
        for _, row in products.iterrows()
    ]


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "message": "VANTOS backend is running",
    }


# ============================================================
# SEARCH
# ============================================================

@app.post("/search")
@app.post("/api/search")
def search_products_api(
    request: SearchRequest,
):

    try:

        search_result = search_products(
            request.query,
            request.top_k,
        )

        results = []

        for result in search_result["results"]:

            product = result.payload

            results.append({

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

                "rating": float(
                    product["rating"]
                ),

                "score": float(
                    result.score
                ),

                "image_url": str(
                    product.get(
                        "image_url",
                        "",
                    )
                ),

            })


        return {

            "query": search_result[
                "query"
            ],

            "intent": search_result[
                "intent"
            ],

            "results": results,

        }


    except Exception as error:

        print(
            f"Search error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "AI search service is "
                "currently unavailable"
            ),
        )


# ============================================================
# INTENT
# ============================================================

@app.post("/api/intent")
def extract_intent_api(
    request: IntentRequest,
):

    try:

        intent = extract_intent(
            request.query
        )

        return {

            "query": request.query,

            "intent": intent,

        }


    except Exception as error:

        print(
            f"Intent error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Intent extraction service "
                "is currently unavailable"
            ),
        )


# ============================================================
# PRODUCT DETAILS
# ============================================================

@app.get("/products/{product_id}")
@app.get("/api/products/{product_id}")
def get_product_api(
    product_id: str,
):

    product = df[
        df["product_id"].astype(str)
        == str(product_id)
    ]

    if product.empty:

        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    return serialize_product(
        product.iloc[0]
    )


# ============================================================
# CATALOG / CATEGORY BROWSING
# ============================================================

@app.get("/products")
@app.get("/api/products")
def list_products(
    category: Optional[str] = None,
    limit: int = 1000,
):

    limit = max(1, min(limit, 1000))

    products = df.copy()

    if category:
        products = products[
            products["category"].astype(str).str.strip().str.lower()
            == category.strip().lower()
        ]

    products = products.head(limit)

    return {
        "count": int(len(products)),
        "category": category,
        "products": serialize_dataframe_products(products),
    }


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.get("/recommendations/{product_id}")
@app.get("/api/recommendations/{product_id}")
def recommendations_api(
    product_id: str,
):

    try:

        recommendations = (
            get_recommendations(
                product_id
            )
        )

        if recommendations is None:

            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )


        # search_service.get_recommendations() already returns
        # serialized product dictionaries for upsell/cross-sell.
        # Do not call .iterrows() on those lists.
        return {

            "product": serialize_product(
                recommendations["product"]
            ),

            "upsell": recommendations.get(
                "upsell", []
            ),

            "cross_sell": recommendations.get(
                "cross_sell", []
            ),

        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            f"Recommendation error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Recommendation service "
                "is currently unavailable"
            ),
        )


# ============================================================
# POST RECOMMENDATION API
# ============================================================

@app.post("/api/recommendations")
def recommendations_post_api(
    request: RecommendationRequest,
):

    return recommendations_api(
        request.product_id
    )


# ============================================================
# UPSELL
# ============================================================

@app.post("/api/upsell")
def upsell_api(
    request: UpsellRequest,
):

    try:
        if request.product_id not in df["product_id"].astype(str).values:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        recommendations = get_upsell_recommendations(
            request.product_id
        )

        return {
            "product_id": request.product_id,
            "upsell": recommendations,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"Upsell error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Upsell service "
                "is currently unavailable"
            ),
        )


# ============================================================
# CROSS-SELL
# ============================================================

@app.post("/api/cross-sell")
def cross_sell_api(
    request: CrossSellRequest,
):

    try:
        if request.product_id not in df["product_id"].astype(str).values:
            raise HTTPException(
                status_code=404,
                detail="Product not found",
            )

        recommendations = get_cross_sell_recommendations(
            request.product_id
        )

        return {
            "product_id": request.product_id,
            "cross_sell": recommendations,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"Cross-sell error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Cross-sell service "
                "is currently unavailable"
            ),
        )


# ============================================================
# UPSELL EXPLANATION
# ============================================================

@app.get(
    "/recommendations/"
    "{product_id}/upsell/"
    "{upsell_product_id}"
)
@app.get(
    "/api/recommendations/"
    "{product_id}/upsell/"
    "{upsell_product_id}"
)
def upsell_explanation_api(
    product_id: str,
    upsell_product_id: str,
):

    try:

        recommendations = (
            get_recommendations(
                product_id
            )
        )

        if recommendations is None:

            raise HTTPException(
                status_code=404,
                detail="Base product not found",
            )


        upsell_product = None


        for product in recommendations[
            "upsell"
        ]:

            if (
                str(
                    product["product_id"]
                )
                == str(
                    upsell_product_id
                )
            ):

                upsell_product = product

                break


        if upsell_product is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Upsell product not found "
                    "for this product"
                ),
            )


        explanation = (
            explain_upsell_with_qwen(
                recommendations[
                    "product"
                ],
                upsell_product,
            )
        )


        return {

            "product_id":
                product_id,

            "upsell_product_id":
                upsell_product_id,

            "explanation":
                explanation,

        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            f"Upsell explanation error: "
            f"{error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Upsell explanation service "
                "is currently unavailable"
            ),
        )


# ============================================================
# EMBEDDING API
# ============================================================

@app.post("/api/embedding")
def embedding_api(
    request: EmbeddingRequest,
):

    try:

        embedding = model.encode(
            request.text,
            normalize_embeddings=True,
        )

        return {

            "text": request.text,

            "dimension":
                len(embedding),

            "embedding":
                embedding.tolist(),

        }


    except Exception as error:

        print(
            f"Embedding error: {error}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Embedding service "
                "is currently unavailable"
            ),
        )