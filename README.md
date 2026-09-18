# jev-local

Local Jev-compatible evaluation server: `POST /v1/systemone` with typed
`noul` / `choice` / `score` questions, probabilities, and confidence.
No waitlist, no API key, no closed weights.

Status: working server + contract tests. Default scorer is deterministic and
carries no intelligence; set `JEVLOCAL_SCORER=hf` for the real frozen-model
logprob scorer.

## Run

```bash
uv venv && uv pip install -e '.[test]'
uv run pytest
uv run uvicorn jevlocal.app:app --port 8000
```

Docker: `docker build -t jev-local . && docker run -p 8000:8000 jev-local`

With a real model (needs GPU + `.[hf]` extra):

```bash
JEVLOCAL_SCORER=hf JEVLOCAL_MODEL=Qwen/Qwen3.5-9B \
  uv run uvicorn jevlocal.app:app --port 8000
```

Open `demo/index.html` in a browser for a minimal UI.

## API

Same shape as the TypeSafe endpoint: `{state, model, questions}` in,
`{model, answers, usage}` out. Limits enforced: choice 1..255 options,
score 2..10 levels, noul criteria keys true/false only.

## Pilot eval (honest, n=24 hand-made items, T=1.0)

| model | accuracy | NLL | ECE(5) | s/question | shuffle flips (4) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.50 | 0.86 | 0.135 | 0.05s | 1 |
| Qwen3-4B | 0.67 | 0.53 | 0.146 | 0.14s | 0 |

Measured on NVIDIA GB10, transformers direct forward pass, per-candidate
mean logprob + softmax, no generation. Pilot only: n=24 cannot support a
go/no-go claim; the decision set needs 150-300 held-out items with CIs.

## Decision set (n=238, balanced, fit/eval split, Wilson 95% CI)

choice/noul/score, 4 departments, templated states, gold labels known.
T fit on fit-half, reported on eval-half.

| model | head | eval acc | 95% CI | NLL | T* |
|---|---|---|---|---|---|
| 0.5B | choice | 0.325 | [0.20, 0.48] | 0.92 | 3.0 |
| 0.5B | noul | 0.875 | [0.74, 0.95] | 0.38 | 0.75 |
| 0.5B | score | 0.410 | [0.27, 0.57] | 1.05 | 1.5 |
| 4B | choice | 0.875 | [0.74, 0.95] | 0.19 | 3.0 |
| 4B | noul | 0.900 | [0.77, 0.96] | 0.19 | 1.0 |
| 4B | score | 0.410 | [0.27, 0.57] | 1.16 | 3.0 |

4B overall eval: 0.73 [0.65, 0.80]. Score head was the weak spot with the
`Level i` verbalizer; variant test on the full score subset (n=78, 4B):
bare digits **0.76** vs `Level i` 0.67 vs label text 0.59. Digits shipped
as default. Default temperatures: choice 1.0, noul 1.0, score 3.0
(score/noul consistent across two independent runs; choice inconsistent,
stays 1.0).

## Drop-in proof

Official `typesafe-sdk==0.6.0` with only `base_url` pointed at our server
(backed by Qwen3-4B on GB10): SSO lockout case returned technical 1.0,
urgent 0.9993, frustration 1.10 in ~0.7s over an SSH tunnel. CORS enabled
for the browser demo (verified in headless Chrome: page load -> fetch ->
200 -> rendered; use 127.0.0.1, not localhost).

## Second set + 8B + error analysis (n=170, refund/moderation/phishing/priority)

| model | head | eval acc | 95% CI |
|---|---|---|---|
| 4B | choice | 0.49 | [0.35, 0.63] |
| 4B | noul | 0.90 | [0.70, 0.97] |
| 4B | score | 0.35 | [0.18, 0.57] |
| 8B | choice | 0.51 | [0.37, 0.65] |
| 8B | noul | 0.85 | [0.64, 0.95] |
| 8B | score | 0.65 | [0.43, 0.82] |
| 8B | set1 overall | 0.86 | [0.78, 0.91] |

8B on set1: choice 0.85, noul 0.90, score 0.82, ECE 0.036, 0.25s/question.

## Model comparison (Sep 2026 benchmarks -> our eval, all <10B, set1 n=119)

Sep-2026 public tables (labellerr, localaimaster, qualiteg) rank Qwen3.5-9B,
Gemma 4 E4B, Phi-4-mini, Qwen3-8B, DeepSeek-R1-Distill-7B on top. Our
decision-set eval (fit/eval split, 95% CI):

| model | eval acc | 95% CI | NLL | ECE | s/q |
|---|---|---|---|---|---|
| Qwen3.5-9B | 0.916 | [0.85, 0.95] | 0.44 | 0.21 | 0.40 |
| Qwen2.5-7B | 0.891 | [0.82, 0.94] | 0.70 | 0.04 | 0.24 |
| Qwen3-8B | 0.857 | [0.78, 0.91] | 0.38 | 0.04 | 0.25 |
| Phi-4-mini (3.8B) | 0.832 | [0.75, 0.89] | 0.54 | 0.07 | 0.13 |
| Qwen2.5-3B | 0.824 | [0.75, 0.88] | 0.40 | 0.06 | 0.12 |
| Mistral-7B-v0.3 | 0.714 | [0.63, 0.79] | 0.52 | 0.09 | 0.23 |

Notes: Gemma gated on HF, skipped (no tokens by policy). Mistral 7B is
outdated, confirms public tables. Default: Qwen3.5-9B for accuracy,
Qwen2.5-3B for speed (0.12s/q, still 0.82).

Error analysis (4B, set2 choice): the model over-predicts middle
categories, every clear approve/reject still lands on review, and the
flag/block boundary is noisy both ways. Stronger boundary wording
("ONLY when genuinely ambiguous...") lifts 0.53 to 0.64 on the same
split. Guidance: write boundary criteria with explicit only-when
constraints; clear cases must name their category.

## Head-to-head vs real Jev (Cloudflare-published jev-1.13.0 outputs)

Same 5 questions asked to published Jev outputs and to our server
(Qwen3.5-9B, GB10). 4/5 agree on the top answer.

| # | question | real Jev | jev-local T=1.0/3.0 (old) | jev-local T=0.5/0.25/0.25 (new) |
|---|---|---|---|---|
| 1 | urgent? payouts failing | 0.95 | 0.9241 | 0.9999 |
| 2 | department? payouts failing | billing 0.87 | technical 0.75 | technical 0.90 (**still differs**) |
| 3 | frustration? payouts failing | 1.04 (Frustrated 0.96) | 1.06 (Frustrated 0.42) | 1.03 (Frustrated 0.96) |
| 4 | department? login/reset | account 1.0 | account 0.97 | account 0.9999 |
| 5a | risk? 5 failed logins + new country | 1.84 (High 0.84) | 1.17 (High 0.39) | 1.62 (High 0.62) |
| 5b | escalate? same | 0.81 | 0.92 | 0.9994 |

Honest read: after per-head temperature fit (choice 0.5, noul 0.25,
score 0.25, NLL-best on set1 eval, Qwen3.5-9B) our sharpness matches
Jev's almost exactly (#3: 0.9593 vs 0.96; #4: 0.9999 vs 1.0).
Direction matches everywhere except #2 (billing vs technical on an
ambiguous payout+failure message; four verbalizer variants V0..V3 all
pick technical, so this is a model prior, not a prompt bug).

## Set3 (~2.6k diverse items, 9B, shipped T, eval-half seed 99)

New domains (support routing, refund approve/review/reject, moderation
allow/flag/block, urgency, phishing, escalation, frustration, priority).
No T fitting here: shipped defaults confirmed as-is.

| head | eval acc | 95% CI | NLL | ECE(5) | n | s/q |
|---|---|---|---|---|---|---|
| choice | 0.818 | [0.788, 0.845] | 0.61 | 0.046 | 704 | 0.46 |
| noul | 0.942 | [0.913, 0.962] | 0.13 | 0.018 | 360 | 0.25 |
| score | 0.714 | [0.656, 0.767] | 0.62 | 0.138 | 252 | 0.38 |
| overall | 0.832 | [0.811, 0.851] | — | — | 1316 | — |

Held-out T check (set2 eval, 9B): NEW {0.5, 0.25, 0.25} beats OLD
{1.0, 1.0, 3.0} on NLL for all three heads at tied accuracy
(choice NLL 0.69->0.60, noul 0.36->0.26, score 0.99->0.86).
Payout #2 root cause: criteria wording owns the routing. With
"billing owns failures" wording the same model picks billing 0.98;
with neutral wording it picks technical. Ambiguous case, not a bug.

## Head post-train decision: not yet

~408 labeled items with repeated templates is too thin for a scorer head;
zero-shot 8B already reaches 0.86 with ECE 0.036. Train a head only after
the set grows past ~2k diverse items.

Known limits: softmax ignores level ordering; option order can shift
logits (0.5B flipped 1/4 on shuffle); usage tokens are estimates.
This is an interface-compatible baseline, not a reproduction of Jev's
undisclosed model or training.
