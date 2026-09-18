"""Scorers: deterministic stub (default) + HF logprob scorer (JEVLOCAL_SCORER=hf).

HF method: identical prompt prefix for all candidates of a question,
teacher-forced mean-logprob over CANDIDATE TOKENS ONLY, softmax with
per-head temperature. Multi-token candidates supported. No generation.
"""

import hashlib
import math

from .models import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion


def _hash01(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _softmax(xs: list[float]) -> list[float]:
    """Fixed T=1 softmax for the stub. Calibration work applies to the HF head."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


class ScorerError(ValueError):
    """Raised for oversized/degenerate inputs. The app maps these to 422."""


MAX_INPUT_TOKENS = 4096


class DeterministicStubScorer:
    """Hash-based stub. Deterministic, valid shapes, no intelligence claimed."""

    model_name = "stub-deterministic-0.1"

    def noul(self, state: str, q: NoulQuestion) -> NoulAnswer:
        p = _hash01(str(state), str(q.instructions))
        return NoulAnswer(type="noul", noul=round(p, 4))

    def choice(self, state: str, q: ChoiceQuestion) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            raise ScorerError("choice criteria must not be empty")
        logits = [_hash01(str(state), str(q.instructions), o) for o in options]
        probs = _softmax(logits)
        best = options[probs.index(max(probs))]
        return ChoiceAnswer(
            type="choice",
            choice=best,
            probabilities={o: round(p, 4) for o, p in zip(options, probs)},
            confidence=round(max(probs), 4),
        )

    def score(self, state: str, q: ScoreQuestion) -> ScoreAnswer:
        n = len(q.criteria)
        if n == 0:
            raise ScorerError("score criteria must not be empty")
        logits = [_hash01(str(state), str(q.instructions), str(i)) for i in range(n)]
        probs = _softmax(logits)
        value = sum(i * p for i, p in enumerate(probs))
        legend = {str(i): level for i, level in enumerate(q.criteria)}
        return ScoreAnswer(
            type="score",
            score=round(value, 4),
            legend=legend,
            probabilities={str(i): round(p, 4) for i, p in enumerate(probs)},
            confidence=round(max(probs), 4),
        )


class HfLogprobScorer:
    """Frozen instruct model + per-candidate mean-logprob scoring.

    Contract mapping (matches docs.typesafe.ai examples):
    - noul: candidates Yes/No, answer = P(Yes). Optional criteria true/false
      descriptions are rendered into the prompt as context.
    - choice: candidates are the bare option keys. Descriptions render as
      `key means: desc` context lines; the demanded answer format is the bare
      key, which is exactly what gets scored.
    - score: candidates are bare level digits `0..n` (variant B wins:
      onyx decision set, Qwen3-4B: digits 0.76 vs 'Level i' 0.67 vs label
      text 0.59). 0-indexed to match the docs example (legend {"0":...},
      score = sum(i * p)). Linear-ordinal assumption.
    - confidence: max-probability, documented baseline.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B",
                 temperatures: dict | None = None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("HfLogprobScorer needs the 'hf' extra: pip install -e '.[hf]'") from e
        self.model_name = model_id
        # Defaults fitted on onyx, Qwen3.5-9B, set1 eval split (seed 99):
        # T sweep 0.25..2.0, NLL-best per head. choice T=0.5 (NLL 0.298,
        # meanmax 0.90); noul T=0.25 (NLL 0.108, meanmax 0.94); score
        # T=0.25 (NLL 0.257, meanmax 0.83). T=1.0 is worse on NLL for
        # all three heads. Confirmed held-out on set2 eval (NEW beats OLD
        # on NLL at tied acc) and on set3 eval-half (n=1316, overall
        # 0.832 [0.811, 0.851]). Verbalizer variants V0..V3 tested on
        # set1+set2 choice-eval: V0 (shipped) best on average, keep.
        # Known wording effect: payout-failure routes technical under
        # neutral criteria, billing when criteria say billing owns
        # failures; ambiguous case, documented in README.
        defaults = {"choice": 0.5, "noul": 0.25, "score": 0.25}
        self.temperatures = defaults | (temperatures or {})
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto")
        self.model.eval()

    def candidate_scores(self, prefix: str, candidates: list[str]) -> tuple[list[float], list[int]]:
        """Mean logprob per candidate over candidate tokens only.

        Tokenization is NOT prefix-preserving at the join, so the offset is
        found by longest common-prefix overlap between tok(prefix) and
        tok(prefix + ' ' + candidate). Degenerate (zero-token) candidates
        raise ScorerError instead of silently scoring 0.0.
        """
        import torch

        if not candidates:
            raise ScorerError("no candidates to score")
        pre_ids = self.tok(prefix, return_tensors="pt").input_ids[0].tolist()
        if len(pre_ids) > MAX_INPUT_TOKENS:
            raise ScorerError(f"prefix too long: {len(pre_ids)} tokens")
        means, lengths = [], []
        with torch.no_grad():
            for c in candidates:
                ids = self.tok(prefix + " " + c, return_tensors="pt")
                full = ids.input_ids[0].tolist()
                if len(full) > MAX_INPUT_TOKENS:
                    raise ScorerError(f"input too long: {len(full)} tokens")
                k = 0
                while k < len(pre_ids) and k < len(full) and pre_ids[k] == full[k]:
                    k += 1
                if k == 0:
                    raise ScorerError("no shared prefix with candidate")
                take = full[k:]
                if not take:
                    raise ScorerError(f"candidate {c!r} scores zero tokens")
                t = ids.input_ids.to(self.model.device)
                mask = ids.attention_mask.to(self.model.device)
                out = self.model(t, attention_mask=mask)
                logp = torch.log_softmax(out.logits[0], dim=-1)
                takes = [logp[i - 1, t[0, i]].item() for i in range(k, len(full))]
                lengths.append(len(takes))
                means.append(sum(takes) / len(takes))
        return means, lengths

    @staticmethod
    def render_prefix(state: str, instructions: str, options: list[str] | None,
                      option_help: dict[str, str] | None,
                      levels: list | None, noul_help: dict[str, str] | None = None) -> str:
        lines = [f"State: {state}", f"Question: {instructions}"]
        if option_help:
            lines += [f"{k} means: {v}" for k, v in option_help.items()]
        if options is not None:
            lines.append("Answer with exactly one of: " + " | ".join(options))
        if noul_help:
            lines += [f"Yes means: {noul_help['true']}", f"No means: {noul_help['false']}"]
        elif levels is not None:
            lines.append("Levels: " + ", ".join(f"{i} = {c}" for i, c in enumerate(levels)))
        lines.append("Answer:")
        return "\n".join(lines)

    def _dist(self, head: str, means: list[float]) -> list[float]:
        from .calibration import softmax
        return softmax(means, self.temperatures[head])

    def noul(self, state: str, q: NoulQuestion) -> NoulAnswer:
        help_ = None
        if q.criteria and q.criteria.get("true") and q.criteria.get("false"):
            help_ = {"true": str(q.criteria["true"]), "false": str(q.criteria["false"])}
        prefix = self.render_prefix(str(state), str(q.instructions), ["Yes", "No"], None, None, help_)
        means, _ = self.candidate_scores(prefix, ["Yes", "No"])
        p = self._dist("noul", means)[0]
        return NoulAnswer(type="noul", noul=round(p, 4))

    def choice(self, state: str, q: ChoiceQuestion) -> ChoiceAnswer:
        options = list(q.criteria.keys())
        if not options:
            raise ScorerError("choice criteria must not be empty")
        help_ = {k: str(v) for k, v in q.criteria.items() if v is not None}
        prefix = self.render_prefix(str(state), str(q.instructions), options, help_ or None, None)
        means, _ = self.candidate_scores(prefix, options)
        probs = self._dist("choice", means)
        best = options[probs.index(max(probs))]
        return ChoiceAnswer(
            type="choice", choice=best,
            probabilities={o: round(p, 4) for o, p in zip(options, probs)},
            confidence=round(max(probs), 4))

    def score(self, state: str, q: ScoreQuestion) -> ScoreAnswer:
        levels = list(q.criteria)
        if not levels:
            raise ScorerError("score criteria must not be empty")
        cands = [str(i) for i in range(len(levels))]
        prefix = self.render_prefix(str(state), str(q.instructions), cands, None, levels)
        means, _ = self.candidate_scores(prefix, cands)
        probs = self._dist("score", means)
        value = sum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            type="score", score=round(value, 4),
            legend={str(i): lv for i, lv in enumerate(levels)},
            probabilities={str(i): round(p, 4) for i, p in enumerate(probs)},
            confidence=round(max(probs), 4))
