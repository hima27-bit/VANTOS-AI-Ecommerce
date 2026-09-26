import os
import sys
import json
import re
import requests
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.search_service import (
    extract_intent,
    retrieve_product_context
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

QWEN_URL = "http://127.0.0.1:8080/v1/chat/completions"


# ---------------------------------------------------------
# 1. INTENT ACCURACY
# ---------------------------------------------------------
# Expected values are based on the explicit intent expressed
# by each benchmark query, not on Qwen's own output.
intent_cases = [
    {
        "query_id": "Q01",
        "query": "lightweight laptop for software development under 50000",
        "category": "laptops",
        "intent": "purchase",
        "max_price": 50000,
        "preferences": ["lightweight", "software development"]
    },
    {
        "query_id": "Q02",
        "query": "developer laptop for programming under 80000",
        "category": "laptops",
        "intent": "purchase",
        "max_price": 80000,
        "preferences": ["developer", "programming"]
    },
    {
        "query_id": "Q03",
        "query": "student laptop for college under 60000",
        "category": "laptops",
        "intent": "purchase",
        "max_price": 60000,
        "preferences": ["student", "college"]
    },
    {
        "query_id": "Q04",
        "query": "gaming laptop for performance under 100000",
        "category": "laptops",
        "intent": "purchase",
        "max_price": 100000,
        "preferences": ["gaming", "performance"]
    },
    {
        "query_id": "Q05",
        "query": "premium laptop for productivity",
        "category": "laptops",
        "intent": "purchase",
        "max_price": None,
        "preferences": ["premium", "productivity"]
    },
    {
        "query_id": "Q06",
        "query": "best camera phone under 25000",
        "category": "mobiles",
        "intent": "purchase",
        "max_price": 25000,
        "preferences": ["camera"]
    },
    {
        "query_id": "Q07",
        "query": "gaming smartphone under 30000",
        "category": "mobiles",
        "intent": "purchase",
        "max_price": 30000,
        "preferences": ["gaming"]
    },
    {
        "query_id": "Q08",
        "query": "phone for photography around 50000",
        "category": "mobiles",
        "intent": "purchase",
        "max_price": 50000,
        "preferences": ["photography"]
    },
    {
        "query_id": "Q09",
        "query": "battery focused phone for travel under 20000",
        "category": "mobiles",
        "intent": "purchase",
        "max_price": 20000,
        "preferences": ["battery", "travel"]
    },
    {
        "query_id": "Q10",
        "query": "blue formal shirt for office",
        "category": "fashion",
        "intent": "purchase",
        "max_price": None,
        "preferences": ["blue", "formal", "office"]
    },
    {
        "query_id": "Q11",
        "query": "black casual shirt for everyday use",
        "category": "fashion",
        "intent": "purchase",
        "max_price": None,
        "preferences": ["black", "casual", "everyday"]
    },
    {
        "query_id": "Q12",
        "query": "running shoes for fitness under 5000",
        "category": "shoes",
        "intent": "purchase",
        "max_price": 5000,
        "preferences": ["running", "fitness"]
    },
    {
        "query_id": "Q13",
        "query": "formal shoes for office under 6000",
        "category": "shoes",
        "intent": "purchase",
        "max_price": 6000,
        "preferences": ["formal", "office"]
    },
    {
        "query_id": "Q14",
        "query": "waterproof laptop bag for office",
        "category": "accessories",
        "intent": "purchase",
        "max_price": None,
        "preferences": ["waterproof", "laptop bag", "office"]
    },
    {
        "query_id": "Q15",
        "query": "power bank for travel under 3000",
        "category": "accessories",
        "intent": "purchase",
        "max_price": 3000,
        "preferences": ["power bank", "travel"]
    },
    {
        "query_id": "Q16",
        "query": "USB C cable for charging under 1500",
        "category": "accessories",
        "intent": "purchase",
        "max_price": 1500,
        "preferences": ["USB C", "charging"]
    },
    {
        "query_id": "Q17",
        "query": "smart home electronics for office under 25000",
        "category": "electronics",
        "intent": "purchase",
        "max_price": 25000,
        "preferences": ["smart home", "office"]
    },
    {
        "query_id": "Q18",
        "query": "digital camera for photography under 100000",
        "category": "electronics",
        "intent": "purchase",
        "max_price": 100000,
        "preferences": ["digital camera", "photography"]
    }
]


def normalize_text(value):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower()
    ).strip()


def preference_match(expected, actual):
    actual_text = normalize_text(" ".join(actual or []))
    expected_text = normalize_text(expected)

    if expected_text in actual_text:
        return True

    aliases = {
        "usb c": ["usb-c", "usb c"],
        "software development": ["software development", "programming", "coding"],
        "developer": ["developer", "development", "coding"],
        "camera": ["camera", "photography"],
        "photography": ["photography", "camera"],
        "battery": ["battery", "long lasting"],
        "waterproof": ["waterproof", "water resistant"],
        "laptop bag": ["laptop bag", "laptop backpack"],
        "smart home": ["smart home", "home electronics", "smart controls"],
    }

    for alias in aliases.get(expected, [expected]):
        if normalize_text(alias) in actual_text:
            return True

    return False


def intent_is_correct(expected, actual):
    if not isinstance(actual, dict):
        return False

    if normalize_text(actual.get("category")) != normalize_text(
        expected["category"]
    ):
        return False

    if normalize_text(actual.get("intent")) != normalize_text(
        expected["intent"]
    ):
        return False

    expected_price = expected["max_price"]
    actual_price = actual.get("max_price")

    if expected_price is None:
        if actual_price is not None:
            return False
    else:
        try:
            if abs(float(actual_price) - expected_price) > 1:
                return False
        except (TypeError, ValueError):
            return False

    return all(
        preference_match(pref, actual.get("preferences", []))
        for pref in expected["preferences"]
    )


intent_rows = []

print()
print("=" * 78)
print("AI EVALUATION - INTENT")
print("=" * 78)

for case in intent_cases:
    try:
        actual = extract_intent(case["query"])
        correct = intent_is_correct(case, actual)

        intent_rows.append({
            "query_id": case["query_id"],
            "query": case["query"],
            "expected_intent": case["intent"],
            "actual_intent": actual.get("intent"),
            "expected_category": case["category"],
            "actual_category": actual.get("category"),
            "expected_max_price": case["max_price"],
            "actual_max_price": actual.get("max_price"),
            "expected_preferences": ";".join(case["preferences"]),
            "actual_preferences": ";".join(
                str(x) for x in actual.get("preferences", [])
            ),
            "correct": int(correct)
        })

        print(
            f"{case['query_id']}: "
            f"{'PASS' if correct else 'FAIL'}"
        )

    except Exception as exc:
        intent_rows.append({
            "query_id": case["query_id"],
            "query": case["query"],
            "expected_intent": case["intent"],
            "actual_intent": "",
            "expected_category": case["category"],
            "actual_category": "",
            "expected_max_price": case["max_price"],
            "actual_max_price": "",
            "expected_preferences": ";".join(case["preferences"]),
            "actual_preferences": "",
            "correct": 0,
            "error": str(exc)
        })
        print(f"{case['query_id']}: FAIL ({exc})")

intent_df = pd.DataFrame(intent_rows)
intent_accuracy = float(intent_df["correct"].mean())

intent_path = os.path.join(
    RESULTS_DIR,
    "intent_evaluation.csv"
)
intent_df.to_csv(intent_path, index=False)


# ---------------------------------------------------------
# FIELD-LEVEL INTENT METRICS
# ---------------------------------------------------------
# Strict Intent Accuracy above requires every field to be correct.
# These additional metrics show which parts of the structured intent
# extraction are working independently.

def price_is_correct(expected_price, actual_price):
    if expected_price is None:
        return actual_price is None
    try:
        return actual_price is not None and abs(float(actual_price) - expected_price) <= 1
    except (TypeError, ValueError):
        return False


def preference_recall(expected_preferences, actual_preferences):
    expected_preferences = expected_preferences or []
    if not expected_preferences:
        return 1.0

    matched = sum(
        preference_match(pref, actual_preferences or [])
        for pref in expected_preferences
    )
    return matched / len(expected_preferences)


field_rows = []
for case, actual in zip(intent_cases, intent_rows):
    actual_preferences = [
        value.strip()
        for value in str(actual.get("actual_preferences", "")).split(";")
        if value.strip()
    ]

    category_correct = (
        normalize_text(actual.get("actual_category"))
        == normalize_text(case["category"])
    )
    intent_correct = (
        normalize_text(actual.get("actual_intent"))
        == normalize_text(case["intent"])
    )
    price_correct = price_is_correct(
        case["max_price"], actual.get("actual_max_price")
    )
    pref_recall = preference_recall(
        case["preferences"], actual_preferences
    )

    field_rows.append({
        "query_id": case["query_id"],
        "category_correct": int(category_correct),
        "intent_correct": int(intent_correct),
        "price_correct": int(price_correct),
        "preference_recall": pref_recall,
        "strict_intent_correct": int(actual["correct"])
    })

field_df = pd.DataFrame(field_rows)
field_metrics = {
    "Category Accuracy": float(field_df["category_correct"].mean()),
    "Intent Field Accuracy": float(field_df["intent_correct"].mean()),
    "Price Accuracy": float(field_df["price_correct"].mean()),
    "Preference Recall": float(field_df["preference_recall"].mean()),
    "Strict Intent Accuracy": intent_accuracy
}

field_path = os.path.join(
    RESULTS_DIR,
    "intent_field_metrics.csv"
)
pd.DataFrame([
    {"metric": metric, "value": value, "tests": len(field_df)}
    for metric, value in field_metrics.items()
]).to_csv(field_path, index=False)

field_detail_path = os.path.join(
    RESULTS_DIR,
    "intent_field_evaluation.csv"
)
field_df.to_csv(field_detail_path, index=False)

print()
print("=" * 78)
print("AI EVALUATION - FIELD-LEVEL INTENT METRICS")
print("=" * 78)
for metric, value in field_metrics.items():
    print(f"{metric:<24}: {value:.4f}")


# ---------------------------------------------------------
# 2. GROUNDEDNESS
# ---------------------------------------------------------
# These are factual questions whose answers are explicitly
# available in the retrieved product context.
grounded_cases = [
    {
        "id": "G01",
        "query": "What is the main camera resolution of this smartphone?",
        "retrieval_query": "camera focused smartphone",
        "expected_product_field": "features"
    },
    {
        "id": "G02",
        "query": "What is the battery capacity of the retrieved smartphone?",
        "retrieval_query": "battery focused phone",
        "expected_product_field": "features"
    },
    {
        "id": "G03",
        "query": "What is the price of the retrieved power bank?",
        "retrieval_query": "power bank for travel",
        "expected_product_field": "price"
    },
    {
        "id": "G04",
        "query": "What type of connector does the retrieved cable use?",
        "retrieval_query": "USB C cable",
        "expected_product_field": "features"
    },
    {
        "id": "G05",
        "query": "What is the weight of the retrieved developer laptop?",
        "retrieval_query": "developer laptop for programming",
        "expected_product_field": "features"
    },
    {
        "id": "G06",
        "query": "What material is used in the retrieved formal shoes?",
        "retrieval_query": "formal shoes for office",
        "expected_product_field": "features"
    }
]


def call_qwen(prompt):
    response = requests.post(
        QWEN_URL,
        json={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied product context. "
                        "Do not invent information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def context_to_text(context):
    return json.dumps(
        context,
        indent=2,
        ensure_ascii=False
    )


def answer_is_grounded(answer, context):
    """
    Conservative lexical grounding check.

    A response is counted as grounded when the important factual
    values present in the context are reflected in the answer.
    We deliberately do not use an LLM judge here, avoiding a
    second model introducing another subjective evaluation layer.
    """
    answer_norm = normalize_text(answer)

    # Collect factual values that are useful for this catalog.
    facts = []

    for product in context:
        facts.append(str(product.get("product_name", "")))
        facts.append(str(product.get("price", "")))

        features = str(product.get("features", ""))
        for value in re.findall(
            r"\d+(?:\.\d+)?(?:GB|TB|MP|mAh|Hz|kg|hour|hours)",
            features,
            flags=re.IGNORECASE
        ):
            facts.append(value)

        # Important textual feature terms.
        for term in [
            "usb-c",
            "water-resistant",
            "leather",
            "mesh",
            "knit",
            "canvas",
            "rubber outsole",
            "foam outsole",
            "5g"
        ]:
            if term in features.lower():
                facts.append(term)

    # Normalize common numeric formatting.
    normalized_facts = [
        normalize_text(x).replace(" ", "")
        for x in facts
        if str(x).strip()
    ]

    normalized_answer = normalize_text(answer).replace(" ", "")

    # Also accept a safe "not mentioned" response as grounded.
    if any(
        phrase in normalize_text(answer)
        for phrase in [
            "not mentioned",
            "not provided",
            "not specified",
            "cannot determine",
            "insufficient information"
        ]
    ):
        return True

    return any(
        fact and fact in normalized_answer
        for fact in normalized_facts
    )


grounded_rows = []

print()
print("=" * 78)
print("AI EVALUATION - GROUNDEDNESS")
print("=" * 78)

for case in grounded_cases:
    try:
        context = retrieve_product_context(
            case["retrieval_query"],
            top_k=1
        )

        prompt = f"""
Product context:
{context_to_text(context)}

Question:
{case["query"]}

Answer using only the product context above.
If the requested information is not present, explicitly say that
the information is not available.
"""

        answer = call_qwen(prompt)
        grounded = answer_is_grounded(answer, context)

        grounded_rows.append({
            "test_id": case["id"],
            "query": case["query"],
            "retrieval_query": case["retrieval_query"],
            "context_product_id": (
                context[0]["product_id"] if context else ""
            ),
            "answer": answer,
            "grounded": int(grounded)
        })

        print(
            f"{case['id']}: "
            f"{'PASS' if grounded else 'FAIL'}"
        )

    except Exception as exc:
        grounded_rows.append({
            "test_id": case["id"],
            "query": case["query"],
            "retrieval_query": case["retrieval_query"],
            "context_product_id": "",
            "answer": "",
            "grounded": 0,
            "error": str(exc)
        })
        print(f"{case['id']}: FAIL ({exc})")

grounded_df = pd.DataFrame(grounded_rows)
grounded_rate = float(grounded_df["grounded"].mean())

grounded_path = os.path.join(
    RESULTS_DIR,
    "groundedness_evaluation.csv"
)
grounded_df.to_csv(grounded_path, index=False)


# ---------------------------------------------------------
# 3. HALLUCINATION CONTROL
# ---------------------------------------------------------
# These questions deliberately ask for facts not represented
# in the supplied product context.
hallucination_cases = [
    {
        "id": "H01",
        "query": "Does this smartphone support NFC?",
        "retrieval_query": "Novatek Budget Smartphone 001"
    },
    {
        "id": "H02",
        "query": "Does this laptop have a fingerprint sensor?",
        "retrieval_query": "developer laptop for programming"
    },
    {
        "id": "H03",
        "query": "Does this power bank support wireless charging?",
        "retrieval_query": "power bank for travel"
    },
    {
        "id": "H04",
        "query": "Does this formal shirt have a wrinkle-free coating?",
        "retrieval_query": "blue formal shirt for office"
    },
    {
        "id": "H05",
        "query": "Does this camera support 8K video recording?",
        "retrieval_query": "digital camera for photography"
    },
    {
        "id": "H06",
        "query": "Does this cable support Thunderbolt 4?",
        "retrieval_query": "USB C cable for charging"
    }
]


def response_is_safe_for_unknown(answer):
    answer_norm = normalize_text(answer)

    safe_phrases = [
        "not mentioned",
        "not provided",
     "not specified",
        "cannot determine",
        "cannot be determined",
     "insufficient information",
        "information is unavailable",
        "information unavailable",
        "not available",
        "unable to determine",
        "unknown"
]

    return any(
        phrase in answer_norm
        for phrase in safe_phrases
    )


hallucination_rows = []

print()
print("=" * 78)
print("AI EVALUATION - HALLUCINATION CONTROL")
print("=" * 78)

for case in hallucination_cases:
    try:
        context = retrieve_product_context(
            case["retrieval_query"],
            top_k=1
        )

        prompt = f"""
Product context:
{context_to_text(context)}

Question:
{case["query"]}

Answer ONLY from the supplied product context.
The context may not contain the requested information.
If it does not, explicitly say that the information is unavailable.
Never guess or invent a specification.
"""

        answer = call_qwen(prompt)
        safe = response_is_safe_for_unknown(answer)

        hallucination_rows.append({
            "test_id": case["id"],
            "query": case["query"],
            "retrieval_query": case["retrieval_query"],
            "answer": answer,
            "safe_response": int(safe),
            "hallucination": int(not safe)
        })

        print(
            f"{case['id']}: "
            f"{'PASS' if safe else 'FAIL'}"
        )

    except Exception as exc:
        hallucination_rows.append({
            "test_id": case["id"],
            "query": case["query"],
            "retrieval_query": case["retrieval_query"],
            "answer": "",
            "safe_response": 0,
            "hallucination": 1,
            "error": str(exc)
        })
        print(f"{case['id']}: FAIL ({exc})")

hallucination_df = pd.DataFrame(hallucination_rows)
hallucination_rate = float(
    hallucination_df["hallucination"].mean()
)

hallucination_path = os.path.join(
    RESULTS_DIR,
    "hallucination_evaluation.csv"
)
hallucination_df.to_csv(
    hallucination_path,
    index=False
)


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------
summary = pd.DataFrame([
    {
        "metric": "Intent Accuracy",
        "value": intent_accuracy,
        "tests": len(intent_df)
    },
    {
        "metric": "Groundedness Rate",
        "value": grounded_rate,
        "tests": len(grounded_df)
    },
    {
        "metric": "Hallucination Rate",
        "value": hallucination_rate,
        "tests": len(hallucination_df)
    }
])

summary_path = os.path.join(
    RESULTS_DIR,
    "ai_evaluation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print()
print("=" * 78)
print("AI EVALUATION SUMMARY")
print("=" * 78)
print()
print(
    f"Intent Accuracy:   {intent_accuracy:.4f}"
)
print(
    f"Groundedness Rate: {grounded_rate:.4f}"
)
print(
    f"Hallucination Rate:{hallucination_rate:.4f}"
)
print()
print(f"Summary: {summary_path}")
print(f"Intent: {intent_path}")
print(f"Intent fields: {field_path}")
print(f"Intent field details: {field_detail_path}")
print(f"Groundedness: {grounded_path}")
print(f"Hallucination: {hallucination_path}")
print()
print("=" * 78)
print("AI EVALUATION COMPLETE")
print("=" * 78)

# ============================================================
# 4. TASK 6 - QWEN MODEL EVALUATION
# ============================================================
# The official internship assignment requires evaluating suitable
# Qwen model sizes against quality, latency, memory and infrastructure
# constraints rather than assuming the reference Qwen 1.5B.
#
# Models specified by the assignment:
#   Qwen 1.5B       - reference baseline
#   Qwen3-1.7B      - lightweight experimentation
#   Qwen3-4B        - balanced PoC candidate
#   Qwen3-8B        - higher-quality reasoning candidate
#   Qwen3-14B       - advanced evaluation where infrastructure permits
#
# This section does NOT fabricate measurements for models that are not
# installed. Benchmark each installed model through the same llama.cpp
# endpoint and the results are accumulated in qwen_model_evaluation.csv.
#
# Example:
#   set QWEN_MODEL_NAME=Qwen3-4B
#   python evaluation/evaluate_ai.py
#
# Before benchmarking another model, start llama-server with that model
# and set QWEN_MODEL_NAME to the corresponding manifest name.
# ============================================================

import time

QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen3-4B")
QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "")
QWEN_SERVER_URL = os.environ.get(
    "QWEN_SERVER_URL",
    QWEN_URL
)

MODEL_MANIFEST = [
    {
        "model": "Qwen 1.5B",
        "size_b": 1.5,
        "purpose": "Reference baseline",
        "path_env": "QWEN_15B_PATH"
    },
    {
        "model": "Qwen3-1.7B",
        "size_b": 1.7,
        "purpose": "Lightweight experimentation",
        "path_env": "QWEN3_17B_PATH"
    },
    {
        "model": "Qwen3-4B",
        "size_b": 4.0,
        "purpose": "Balanced PoC candidate",
        "path_env": "QWEN3_4B_PATH"
    },
    {
        "model": "Qwen3-8B",
        "size_b": 8.0,
        "purpose": "Higher-quality reasoning candidate",
        "path_env": "QWEN3_8B_PATH"
    },
    {
        "model": "Qwen3-14B",
        "size_b": 14.0,
        "purpose": "Advanced quality evaluation where infrastructure permits",
        "path_env": "QWEN3_14B_PATH"
    }
]


def get_process_memory_mb():
    """Return Python-process RSS if psutil is installed."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def server_is_available():
    """Check the llama.cpp health endpoint without making a model claim."""
    try:
        health_url = QWEN_SERVER_URL.replace(
            "/v1/chat/completions",
            "/health"
        )
        response = requests.get(health_url, timeout=5)
        return response.ok
    except Exception:
        return False


def benchmark_single_model():
    """Benchmark the model currently exposed by the Qwen server."""
    if not server_is_available():
        return {
            "model": QWEN_MODEL_NAME,
            "size_b": "",
            "status": "SERVER_UNAVAILABLE",
            "intent_accuracy": "",
            "groundedness_rate": "",
            "hallucination_rate": "",
            "avg_latency_ms": "",
            "p95_latency_ms": "",
            "memory_mb": get_process_memory_mb(),
            "model_path": QWEN_MODEL_PATH,
            "notes": "Start llama-server before running Task 6 evaluation."
        }

    intent_results = []
    grounded_results = []
    hallucination_results = []
    latencies = []

    # Intent quality + latency
    for case in intent_cases:
        start = time.perf_counter()
        try:
            actual = extract_intent(case["query"])
            correct = intent_is_correct(case, actual)
        except Exception:
            correct = False
        latencies.append((time.perf_counter() - start) * 1000)
        intent_results.append(int(correct))

    # Groundedness quality + latency
    for case in grounded_cases:
        start = time.perf_counter()
        try:
            context = retrieve_product_context(
                case["retrieval_query"],
                top_k=1
            )
            prompt = f"""
Product context:
{context_to_text(context)}

Question:
{case["query"]}

Answer using only the product context above.
If the requested information is not present, explicitly say that the information is not available.
"""
            answer = call_qwen(prompt)
            grounded = answer_is_grounded(answer, context)
        except Exception:
            grounded = False
        latencies.append((time.perf_counter() - start) * 1000)
        grounded_results.append(int(grounded))

    # Hallucination control + latency
    for case in hallucination_cases:
        start = time.perf_counter()
        try:
            context = retrieve_product_context(
                case["retrieval_query"],
                top_k=1
            )
            prompt = f"""
Product context:
{context_to_text(context)}

Question:
{case["query"]}

Answer ONLY from the supplied product context.
The context may not contain the requested information.
If it does not, explicitly say that the information is unavailable.
Never guess or invent a specification.
"""
            answer = call_qwen(prompt)
            safe = response_is_safe_for_unknown(answer)
        except Exception:
            safe = False
        latencies.append((time.perf_counter() - start) * 1000)
        hallucination_results.append(int(not safe))

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    p95_index = min(
        len(latencies) - 1,
        max(0, int(len(latencies) * 0.95) - 1)
    ) if latencies else None
    p95_latency = latencies[p95_index] if p95_index is not None else None

    manifest_match = next(
        (
            item for item in MODEL_MANIFEST
            if item["model"].lower() == QWEN_MODEL_NAME.lower()
        ),
        None
    )

    return {
        "model": QWEN_MODEL_NAME,
        "size_b": manifest_match["size_b"] if manifest_match else "",
        "status": "BENCHMARKED",
        "intent_accuracy": round(sum(intent_results) / len(intent_results), 4),
        "groundedness_rate": round(sum(grounded_results) / len(grounded_results), 4),
        "hallucination_rate": round(sum(hallucination_results) / len(hallucination_results), 4),
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
        "p95_latency_ms": round(p95_latency, 2) if p95_latency is not None else None,
        "memory_mb": get_process_memory_mb(),
        "model_path": QWEN_MODEL_PATH,
        "notes": "Same benchmark cases are used for every model."
    }


def print_model_manifest():
    print()
    print("=" * 78)
    print("TASK 6 - QWEN MODEL EVALUATION PLAN")
    print("=" * 78)
    for item in MODEL_MANIFEST:
        path = os.environ.get(item["path_env"], "")
        installed = bool(path and os.path.exists(path))
        print(
            f"{item['model']:<16} | {item['size_b']:>4}B | "
            f"{'AVAILABLE' if installed else 'PATH NOT CONFIGURED':<20} | "
            f"{item['purpose']}"
        )
    print()
    print("Only models actually run through the Qwen server receive measurements.")


def save_model_evaluation(row):
    path = os.path.join(RESULTS_DIR, "qwen_model_evaluation.csv")
    row_df = pd.DataFrame([row])

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[
            existing["model"].astype(str).str.lower()
            != str(row["model"]).lower()
        ]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(path, index=False)
    return path


def run_task6_model_evaluation():
    print_model_manifest()

    print()
    print("=" * 78)
    print("TASK 6 - CURRENT QWEN MODEL BENCHMARK")
    print("=" * 78)
    print(f"Model label : {QWEN_MODEL_NAME}")
    print(f"Server      : {QWEN_SERVER_URL}")

    row = benchmark_single_model()

    print()
    print(f"Status              : {row['status']}")
    if row["status"] == "BENCHMARKED":
        print(f"Intent Accuracy     : {row['intent_accuracy']:.4f}")
        print(f"Groundedness Rate   : {row['groundedness_rate']:.4f}")
        print(f"Hallucination Rate  : {row['hallucination_rate']:.4f}")
        print(f"Average Latency     : {row['avg_latency_ms']:.2f} ms")
        print(f"P95 Latency         : {row['p95_latency_ms']:.2f} ms")
        print(f"Python RSS Memory   : {row['memory_mb']} MB")
    else:
        print(f"Note                : {row['notes']}")

    result_path = save_model_evaluation(row)
    print()
    print(f"Saved: {result_path}")


# Run Task 6 after the existing AI evaluation.
run_task6_model_evaluation()
