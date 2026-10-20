# Testing the agents

There are four kinds of test. Run them in this order; each answers a different question.

| # | Test | Question it answers | Needs AWS | Time |
|---|------|---------------------|-----------|------|
| 1 | Offline unit tests | Is the code correct? (routing logic, tool loop, reranking, scoring) | no | ~5 s |
| 2 | Live smoke | Do the real agents route and answer sensibly right now? | yes | ~30 s |
| 3 | Evaluation | How good is the whole system, as a number I can compare over time? | yes | ~6 min (3 runs) |
| 4 | Calibration | Is each KB's `min_score` still right after a KB or content change? | yes | ~1 min |

All commands run from the project root in PowerShell and use the project's `.venv`.

## 0. One-time setup

```powershell
cd D:\Projects\imazeemahmad\ai-agent-aws-traditional-approach
python -m venv .venv                       # skip if .venv already exists
.\.venv\Scripts\pip install -r requirements.txt
```

For tests 2-4 log in first (credentials last about an hour):

```powershell
saml2aws login --profile sand
```

The KB IDs, models and thresholds come from `config.yaml`. Your SAML role needs `bedrock:Retrieve` on both KBs,
`bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the chat models, and `bedrock:Rerank` plus
`bedrock:InvokeModel` on Cohere Rerank 3.5 (see `docs/knowledge-base-setup.md`).

## 1. Offline unit tests (no AWS)

```powershell
.\tests\scripts\run_tests.ps1                                       # via the script
python -m unittest discover -s tests -t . -v          # or directly
python -m unittest tests.test_routing -v              # one file
python -m unittest tests.test_routing.RoutingTests.test_runaway_tool_calls_are_stopped   # one test
```

**Expected:** `Ran 27 tests ... OK`. These use fake KBs and a fake model, so they never touch AWS and are not affected by
KB content. Run them after **every code change**.

| File | Tests | What it protects |
|------|-------|------------------|
| `test_routing.py` | 9 | refusal with no LLM call; pure price question routed straight to the price agent; exposure answer with the tool on offer; passages kept out of the saved history; the tool loop and agent labels; unique `[P#]` ids across several tool calls; runaway tool calls stopped; behaviour without the price agent; follow-up query rewrite |
| `test_retrieval.py` | 7 | fetch, rerank, then filter on the rerank score; irrelevant query returns nothing; `min_score=0` for tuning; managed vs vector KB auto-detection; plain vector mode when reranking is off; unrelated errors not swallowed; placeholder KB id rejected |
| `test_evaluate.py` | 11 | the case file is well formed; every scoring check (route, answer, speculation guard, figure grounding, refusals, history numbers); the regression for streamed chunks being joined with spaces |

If a test fails, the assertion message says which behaviour broke. Fix the code, not the test, unless the behaviour was
changed on purpose.

## 2. Live smoke (real agents, real KBs)

```powershell
.\tests\scripts\run_tests.ps1 live                                              # one question per route
.\tests\scripts\run_tests.ps1 live "How much does Fazkurooni cost?" "Who won the AFL?"
.\tests\scripts\run_tests.ps1 live -Model nova                                  # another configured model
```

Directly: `python -m tests.live_smoke [--model nova] [--profile sand] ["question" ...]`. Nothing is saved to `sessions/`.

For each question it prints the labelled reply, who answered, the sources with their agent and relevance score, any
sub-agent call (question sent) and the time. What the four default questions should show:

| Question | Expected |
|----------|----------|
| Who won the AFL grand final? | `[system]` refusal; sources from: none; about 2 s (no LLM call) |
| Price of the Fa Inni Qareeb artwork | "routed directly to the price agent"; `[price agent]`; `$90`; sources from: price |
| Stages of commissioning at Barangaroo | `[main agent]`; six stages with `[1]` citations; sources from: main |
| Barangaroo stages **and** the price of Fazkurooni | should show both agents. **With the current tool-choice design the price agent is not called** (known failure, see the baseline) |

Use this to look at an answer yourself; it is not a pass/fail gate. For that, use the evaluation.

## 3. Evaluation (scored)

```powershell
.\tests\scripts\run_tests.ps1 eval -Label my-change -Repeat 3          # all 23 cases, 3 runs each
.\tests\scripts\run_tests.ps1 eval -Label both-only -Filter both       # one category or one case id
python main.py --eval-compare eval/results/<A>.json eval/results/<B>.json
```

Use `-Repeat 3` or more: LLM answers vary, and a flaky case is a finding in itself. Output per case:
`PASS` (all runs), `FLAKY` (some runs), `FAIL` (none), then route expected -> observed, latency, LLM calls, and the
reasons a case failed. The summary gives overall pass rate; route, answer and grounded accuracy; average latency, LLM
calls and retrievals (cost); and pass rate by category. Each run is saved to `eval/results/<time>-<label>.json`.

The three checks per case (all deterministic, no LLM judge):

| Check | Passes when |
|-------|-------------|
| route | the agents that supplied sources match `route` in `eval/cases.yaml`: `exposure`, `price`, `both` or `none` |
| answer | every `must_include` fact is present, no `must_not_include` text is present, and `refuse: true` cases say they cannot answer |
| grounded | every amount, percentage and k/M figure in the answer appears in a retrieved passage, the question or the history |

**Baseline of the current design (tool choice, 2026-10-19):** overall 78%; exposure 100%, price 80%, **both 0%**, out of
scope 100%, follow-up 100%; route 83%, answer 78%, grounded 100%; 3.9 s, 0.9 LLM calls. Any new design must beat this
without regressing a category that already passes. Compare with `--eval-compare`: it lists every case as FIXED or
REGRESSED.

### When to run it
- After any change to prompts, routing, models, `min_score` or reranking.
- After changing KB content or re-syncing: first update the expectations in `eval/cases.yaml`, because the expected
  facts come from the KB text.
- Before and after every architecture change, with the **same** `-Repeat` and cases so the numbers are comparable.

### Adding or changing a case
Every real failure seen in a chat session should become a case. Add it to `eval/cases.yaml`:

```yaml
- id: price-06
  category: price               # exposure | price | both | out_of_scope | followup
  question: How much is the Iqra calligraphy artwork?
  route: price                  # exposure | price | both | none
  must_include: ["$189"]        # literal, case-insensitive; "re:..." = regex; a nested list = any one of
  must_not_include: ["re:may represent"]
  # refuse: true                # for out-of-scope cases
  # history: [{user: ..., assistant: ...}]     # for follow-ups
```

Take the expected value from the **source document**, never from the assistant's answer. Run
`python -m unittest tests.test_evaluate` afterwards: it checks the file is well formed.

### Scoring a different architecture
Add a class with `ask(question, history) -> Result` to `TARGETS` in `src/evaluate.py` (for example a router or a
LangGraph version) and run `python main.py --eval --eval-target <name> --eval-label <name> --eval-repeat 3`. The cases
and checks do not change, so the scores compare directly with the baseline.

## 4. Calibration (relevance thresholds)

```powershell
python main.py --calibrate          # uses eval/relevance_set.yaml
```

Prints the best rerank score of every in-scope and out-of-scope question per KB, and a suggested `min_score`. Run it
after changing KB content, the embedding setup or the reranker. `separable` means a threshold exists (copy the
suggestion into `config.yaml`); `OVERLAP` means no threshold separates the questions, so look at the KB content or
chunking, not the number. Extend `eval/relevance_set.yaml` to 10+ questions per side for a trustworthy value. After
changing a threshold, run the evaluation again.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `No virtual environment at ...` | Create `.venv` (step 0). |
| Script blocked by execution policy | `powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\scripts\run_tests.ps1 ...` |
| `ExpiredTokenException` / `UnrecognizedClientException` | `saml2aws login --profile sand`, then re-run. |
| `SSL: CERTIFICATE_VERIFY_FAILED` | The proxy CA bundle is missing. Check `aws.ca_bundle: certs/ca-bundle.pem` in `config.yaml`. |
| `AccessDeniedException` on `Rerank` or `Retrieve` | Add the IAM actions listed in step 0, and enable model access to Cohere Rerank 3.5. |
| `knowledge_base_id is not set` | Put the real 10-character ID in `config.yaml` (or `BEDROCK_KB_ID` / `BEDROCK_PRICE_KB_ID`). |
| `Incompatible configuration ... managed` | Handled automatically: the app detects managed vs vector KBs. If it still appears, the KB id is wrong. |
| `ThrottlingException` during eval | Lower the load: `-Repeat 1`, or run a category with `-Filter`. |
| Offline tests fail after editing `config.yaml` | The tests read the real `config.yaml` (prompts, `max_tool_rounds`); check the keys they use still exist. |
| Eval scores moved but the code did not | Check whether KB content was re-synced, then update `eval/cases.yaml` expectations and re-run `--calibrate`. |
