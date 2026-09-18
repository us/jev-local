import math

import pytest

from jevlocal.calibration import ece, fit_temperature, nll, softmax
from jevlocal.scorer import HfLogprobScorer, ScorerError, default_temperatures


def test_default_temperatures_known_and_fallback():
    assert default_temperatures("Qwen/Qwen3.5-9B", False) == {"choice": 0.5, "noul": 0.25, "score": 0.25}
    assert default_temperatures("Qwen/Qwen3.5-9B", True) == {"choice": 1.0, "noul": 0.25, "score": 0.25}
    assert default_temperatures("Qwen/Qwen2.5-3B-Instruct", True)["choice"] == 1.5
    fb = default_temperatures("Some/Unknown-1B", False)
    assert set(fb) == {"choice", "noul", "score"} and all(t > 0 for t in fb.values())
    fb2 = default_temperatures("Some/Unknown-1B", True)
    assert set(fb2) == {"choice", "noul", "score"}
    # returned dict is a copy: mutating it must not poison the table
    fb["choice"] = 99.0
    assert default_temperatures("Some/Unknown-1B", False)["choice"] == 1.0


def test_maybe_wrap_chat_falls_back_without_template():
    s = HfLogprobScorer.__new__(HfLogprobScorer)
    s.chat = True
    s.tok = type("T", (), {"chat_template": None})()
    assert s.maybe_wrap_chat("hello") == "hello"
    s.chat = False
    assert s.maybe_wrap_chat("hello") == "hello"


def test_softmax_sums_to_one_and_temperature_sharpens():
    assert abs(sum(softmax([1.0, 2.0, 3.0])) - 1) < 1e-9
    flat = softmax([1.0, 2.0], temperature=2.0)
    sharp = softmax([1.0, 2.0], temperature=0.5)
    assert sharp[1] > flat[1] > 0.5
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0)
    with pytest.raises(ValueError):
        softmax([], temperature=1.0)


def test_nll_and_ece():
    assert nll([1.0, 1.0]) == pytest.approx(0.0)
    assert nll([0.5, 0.5]) == pytest.approx(math.log(2))
    assert ece([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert ece([1.0], [False]) == pytest.approx(1.0)
    # multi-bin weighting: two bins, each half weight
    assert ece([0.9, 0.9, 0.1, 0.1], [True, False, False, False]) == pytest.approx(0.25)


def test_fit_temperature_sharpens_confident_gold():
    logits = [[3.0, 0.0], [0.0, 3.0], [2.0, 0.5]]
    gold = [0, 1, 0]
    best_t, best_nll = fit_temperature(logits, gold)
    assert best_t < 1.0  # all-correct input must sharpen, not stay at 1.0
    base = nll([softmax(l)[g] for l, g in zip(logits, gold)])
    assert best_nll <= base
    with pytest.raises(ValueError):
        fit_temperature([], [])


def test_render_prefix_branches():
    p = HfLogprobScorer.render_prefix("s", "q?", ["a", "b"], {"a": "first"}, None)
    assert "a means: first" in p
    assert "Answer with exactly one of: a | b" in p
    assert p.endswith("Answer:")
    p2 = HfLogprobScorer.render_prefix("s", "q?", ["0", "1"], None, ["lo", "hi"])
    assert "0 = lo" in p2 and "Answer with exactly one of: 0 | 1" in p2
    p3 = HfLogprobScorer.render_prefix("s", "q?", ["Yes", "No"], None, None,
                                       {"true": "urgent", "false": "calm"})
    assert "Yes means: urgent" in p3
