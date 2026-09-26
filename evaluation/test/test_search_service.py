import pandas as pd
import pytest

import services.search_service as search_service


def test_normalize_intent_maps_category_and_price():
    raw = {
        "category": "laptop",
        "intent": "purchase",
        "max_price": "₹50,000",
        "preferences": ["lightweight", "lightweight"],
    }

    result = search_service.normalize_intent(raw)

    assert result["category"] == "laptops"
    assert result["intent"] == "purchase"
    assert result["max_price"] == 50000.0
    assert result["preferences"] == ["lightweight"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("lightweight laptop for coding", "laptops"),
        ("best smartphone for photography", "mobiles"),
        ("blue formal shirt for office", "fashion"),
        ("running shoes", "shoes"),
        ("wireless headphones", "accessories"),
    ],
)
def test_category_fallback(query, expected):
    assert search_service.extract_category_from_query(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("laptop under ₹50,000", 50000.0),
        ("phone below 30000", 30000.0),
        ("shoes less than Rs. 5000", 5000.0),
        ("laptop within 45,000", 45000.0),
        ("phone ₹25,000 or less", 25000.0),
    ],
)
def test_price_extraction(query, expected):
    assert search_service.extract_max_price_from_query(query) == expected


def test_price_extraction_returns_none_when_constraint_is_missing():
    assert search_service.extract_max_price_from_query(
        "lightweight laptop"
    ) is None


def test_preference_extraction():
    query = (
        "lightweight laptop for software development "
        "with good battery life"
    )

    preferences = search_service.extract_preferences_from_query(query)

    assert "lightweight" in preferences
    assert "software development" in preferences
    assert "battery" in preferences


def test_search_products_orchestrates_retrieval_and_reranking(monkeypatch):
    class FakeResult:
        def __init__(self, product_id):
            self.payload = {
                "product_id": product_id,
                "product_name": f"Test Product {product_id}",
                "category": "laptops",
                "price": 40000,
            }

    candidates = [
        FakeResult("LAP001"),
        FakeResult("LAP002"),
    ]
    final_results = [FakeResult("LAP002")]

    monkeypatch.setattr(
        search_service,
        "qwen_semantic_search",
        lambda query, top_k=10: (
            {
                "category": "laptops",
                "intent": "purchase",
                "max_price": 50000.0,
                "preferences": ["lightweight"],
            },
            candidates,
        ),
    )

    monkeypatch.setattr(
        search_service,
        "rerank_with_qwen",
        lambda query, intent, results, top_k=5:
            final_results[:top_k],
    )

    response = search_service.search_products(
        "lightweight laptop under ₹50,000",
        top_k=1,
    )

    assert response["query"] == "lightweight laptop under ₹50,000"
    assert response["intent"]["category"] == "laptops"
    assert len(response["results"]) == 1
    assert response["results"][0].payload["product_id"] == "LAP002"


def test_product_lookup_returns_known_product():
    product_id = str(search_service.df.iloc[0]["product_id"])

    product = search_service.get_product_by_id(product_id)

    assert product is not None
    assert product["product_id"] == product_id
    assert "product_name" in product
    assert "category" in product
    assert "price" in product


def test_product_lookup_returns_none_for_unknown_product():
    assert search_service.get_product_by_id("DOES_NOT_EXIST") is None


def test_recommendation_functions_return_lists_for_known_product():
    product_id = str(search_service.df.iloc[0]["product_id"])

    cross_sell = search_service.get_cross_sell_recommendations(product_id)
    upsell = search_service.get_upsell_recommendations(product_id)
    recommendations = search_service.get_recommendations(product_id)

    assert isinstance(cross_sell, list)
    assert isinstance(upsell, list)

    assert recommendations is not None
    assert recommendations["product"]["product_id"] == product_id
    assert isinstance(recommendations["cross_sell"], list)
    assert isinstance(recommendations["upsell"], list)


def test_recommendations_return_none_for_unknown_product():
    assert search_service.get_recommendations("DOES_NOT_EXIST") is None


def test_cross_sell_and_upsell_return_empty_lists_for_unknown_product():
    assert search_service.get_cross_sell_recommendations(
        "DOES_NOT_EXIST"
    ) == []

    assert search_service.get_upsell_recommendations(
        "DOES_NOT_EXIST"
    ) == []
