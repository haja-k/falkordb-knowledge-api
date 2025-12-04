"""Quick test script for the /graphrag endpoint.

Run this while the FalkorDB API is running locally (e.g. uvicorn main:app --reload).
"""
import os
import requests

API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")

def test_sample():
    body = {
        "question": "What are the key stakeholders in Route Location planning?",
        "top_k": 6,
        "hops": 1,
        "labels": None,
        "alpha_vec": 0.6,
        "beta_kw": 0.4,
        "use_mmr": True,
        "use_cross_doc": True
    }
    r = requests.post(f"{API_HOST}/graphrag", json=body)
    print("Status:", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == '__main__':
    test_sample()
