# VANTOS Test Suite

The test suite covers the core service logic and FastAPI endpoints required by the internship project.

## Install

From the project root:

```powershell
pip install pytest httpx
```

If `pytest` is already installed, only install `httpx` if the API tests report that it is missing.

## Run

```powershell
pytest -v
```

## Test groups

- `test_search_service.py`
  - intent normalization
  - category extraction
  - price extraction
  - search orchestration
  - recommendation output structure

- `test_api.py`
  - health endpoint
  - product lookup
  - validation
  - search API
  - intent API
  - recommendation/upsell/cross-sell error handling
  - embedding validation

The search and intent API tests mock the expensive Qwen/semantic-search calls so the test suite remains fast and deterministic.
