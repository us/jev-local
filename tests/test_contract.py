from fastapi.testclient import TestClient

from jevlocal.app import app

client = TestClient(app)

BASE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
}


def _eval(questions, **kw):
    return client.post("/v1/systemone", json={**BASE, "questions": questions, **kw})


def test_happy_path_all_types():
    r = _eval({
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None}},
        "frustration": {"type": "score", "instructions": "How frustrated?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "stub-deterministic-0.1"
    noul = body["answers"]["is_urgent"]
    assert noul["type"] == "noul" and 0 <= noul["noul"] <= 1
    assert "confidence" not in noul
    ch = body["answers"]["department"]
    assert set(ch["probabilities"]) == {"billing", "technical", "sales"}
    assert all(0 <= p <= 1 for p in ch["probabilities"].values())
    assert abs(sum(ch["probabilities"].values()) - 1) < 1e-3
    assert ch["choice"] == max(ch["probabilities"], key=ch["probabilities"].get)
    assert ch["confidence"] == max(ch["probabilities"].values())
    sc = body["answers"]["frustration"]
    assert sc["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(sc["probabilities"]) == {"0", "1", "2"}
    assert all(0 <= p <= 1 for p in sc["probabilities"].values())
    assert abs(sum(sc["probabilities"].values()) - 1) < 1e-3
    assert 0 <= sc["confidence"] <= 1
    assert sc["confidence"] == max(sc["probabilities"].values())
    assert 0 <= sc["score"] <= 2


def test_score_level_bounds():
    for n, ok in [(1, False), (2, True), (10, True), (11, False)]:
        r = _eval({"s": {"type": "score", "instructions": "x", "criteria": ["l"] * n}})
        assert (r.status_code == 200) == ok, (n, r.status_code)


def test_choice_bounds():
    r = _eval({"c": {"type": "choice", "instructions": "x", "criteria": {}}})
    assert r.status_code == 422
    big = {f"o{i}": None for i in range(256)}
    r = _eval({"c": {"type": "choice", "instructions": "x", "criteria": big}})
    assert r.status_code == 422
    for n in (1, 255):
        ok = {f"o{i}": None for i in range(n)}
        r = _eval({"c": {"type": "choice", "instructions": "x", "criteria": ok}})
        assert r.status_code == 200, n


def test_noul_bad_criteria_keys():
    r = _eval({"u": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}})
    assert r.status_code == 422


def test_structured_state_and_criteria():
    for state in ({"message": "charged twice", "tx": [1, 2]}, ["msg1", "msg2"]):
        r = client.post("/v1/systemone", json={
            "state": state, "model": "jev-latest",
            "questions": {"d": {"type": "choice", "instructions": {"q": "route?"},
                                "criteria": {"billing": {"includes": ["Charges"]}, "other": None}}}})
        assert r.status_code == 200, r.text


def test_deterministic_all_types():
    q = {**BASE, "questions": {
        "u": {"type": "noul", "instructions": "Urgent?"},
        "c": {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": None}},
        "s": {"type": "score", "instructions": "Rate?", "criteria": ["lo", "hi"]}}}
    a = client.post("/v1/systemone", json=q).json()
    b = client.post("/v1/systemone", json=q).json()
    assert a == b
