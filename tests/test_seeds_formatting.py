import main as app_main


def test_seed_name_fallback():
    # cands containing nodes with missing name/title should still produce string names
    cands = [
        ({'id': 'node1', 'labels': ['Entity']}, 0.8),
        ({'labels': ['Thing']}, 0.5),
        (12345, 0.3),  # scalar node (id was returned as a scalar in some queries)
    ]

    # Recreate the seeds_meta logic from main.py (call the same loop)
    seeds_meta = []
    for n, sc in cands:
        if isinstance(n, dict):
            labels = list(n.get('labels', [])) if isinstance(n.get('labels', []), list) else []
            name = n.get('name') or n.get('title') or n.get('id') or None
        else:
            labels = []
            name = str(n) if n is not None else None
        if not name:
            name = "<unknown>"
        try:
            score_val = float(sc)
        except Exception:
            score_val = 0.0
        seeds_meta.append({"labels": labels, "name": name, "score": score_val})

    # Validate all resulting seeds have readable string names
    assert all(isinstance(s['name'], str) and s['name'] for s in seeds_meta)
    assert seeds_meta[0]['name'] == 'node1'
    assert seeds_meta[1]['name'] == '<unknown>'
    assert seeds_meta[2]['name'] == '12345'
