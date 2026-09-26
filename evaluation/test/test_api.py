import pytest
from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def get_known_product_id():
    return str(main.df.iloc[0]["product_id"])


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_product_endpoint_returns_known_product():
    product_id = get_known_product_id()

    response = client.get(f"/api/products/{product_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == product_id
    assert "product_name" in data
    assert "category" in data
    assert "price" in data


def test_product_endpoint_returns_404_for_unknown_product():
    response = client.get("/api/products/DOES_NOT_EXIST")

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found"


def test_search_request_validation_rejects_empty_query():
    response = client.post(
        "/api/search",
        json={"query": "", "top_k": 5},
    )

    assert response.status_code == 422


def test_search_request_validation_rejects_invalid_top_k():
    response = client.post(
        "/api/search",
        json={"query": "laptop", "top_k": 0},
    )

    assert response.status_code == 422


def test_search_endpoint_with_mocked_service(monkeypatch):
    class FakeResult:
        def __init__(self):
            self.payload = {
                "product_id": "LAP001",
                "product_name": "Test Laptop",
                "category": "laptops",
                "price": 45000,
                "rating": 4.5,
                "image_url": "",
            }
            self.score = 0.91

    monkeypatch.setattr(
        main,
        "search_products",
        lambda query, top_k: {
            "query": query,
            "intent": {
                "category": "laptops",
                "intent": "purchase",
                "max_price": 50000.0,
                "preferences": ["lightweight"],
            },
            "results": [FakeResult()],
        },
    )

    response = client.post(
        "/api/search",
        json={
            "query": "lightweight laptop under ₹50,000",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "lightweight laptop under ₹50,000"
    assert data["intent"]["category"] == "laptops"
    assert len(data["results"]) == 1
    assert data["results"][0]["product_id"] == "LAP001"


def test_intent_endpoint_with_mocked_service(monkeypatch):
    expected = {
        "category": "laptops",
        "intent": "purchase",
        "max_price": 50000.0,
        "preferences": ["lightweight", "software development"],
    }

    monkeypatch.setattr(main, "extract_intent", lambda query: expected)

    response = client.post(
        "/api/intent",
        json={"query": "lightweight laptop for software development under ₹50,000"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == expected


def test_recommendations_endpoint_returns_404_for_unknown_product():
    response = client.get("/api/recommendations/DOES_NOT_EXIST")

    assert response.status_code == 404


def test_upsell_endpoint_returns_404_for_unknown_product():
    response = client.post(
        "/api/upsell",
        json={"product_id": "DOES_NOT_EXIST"},
    )

    assert response.status_code == 404


def test_cross_sell_endpoint_returns_404_for_unknown_product():
    response = client.post(
        "/api/cross-sell",
        json={"product_id": "DOES_NOT_EXIST"},
    )

    assert response.status_code == 404


def test_embedding_request_validation_rejects_empty_text():
    response = client.post(
        "/api/embedding",
        json={"text": ""},
    )

    assert response.status_code == 422
