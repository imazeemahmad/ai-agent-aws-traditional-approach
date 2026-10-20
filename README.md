# Sydney Art Market Exposure Assistant (POC)

CLI chat assistant on AWS Bedrock (Converse API). Switch LLMs mid-conversation; sessions and context are tracked and saved.

## Environment setup (Windows / PowerShell)

### 1. Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.11+ (developed on 3.13) | `python --version` |
| AWS CLI | v2 | `aws --version` |
| saml2aws | latest | `saml2aws --version` |

Install saml2aws if missing (pick one): `choco install saml2aws`, `scoop install saml2aws`, or download the
Windows binary from the saml2aws GitHub releases page and add it to `PATH`.

### 2. Python virtual environment

```powershell
cd D:\Projects\imazeemahmad\ai-agent-aws-traditional-approach
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 3. One-time saml2aws configuration

```powershell
saml2aws configure     # IdP provider, IdP URL, username (get these from your IT/cloud team)
```

### 4. Log in to AWS (repeat when credentials expire, typically every ~1 hour)

```powershell
saml2aws login --profile sand                  # pick the account/role that has Bedrock access
aws sts get-caller-identity --profile sand     # sanity check: shows account and assumed role
```

The profile name must match `aws.profile` in `config.yaml` (currently `sand`). If you omit `--profile`, saml2aws
writes the `saml` profile. Precedence: `--profile` on `main.py` > `aws.profile` in `config.yaml` > `AWS_PROFILE`
env var. So to use `$env:AWS_PROFILE`, set `aws.profile: null` in `config.yaml`.

Credentials are read when the app starts. If you get `ExpiredTokenException`, run `saml2aws login` again and
restart `main.py` (use `--session <id>` to resume your conversation).

### 5. AWS / Bedrock prerequisites

- **Region**: `config.yaml` uses `us-east-1`. The `us.` model IDs are cross-region inference profiles, so keep the region in the US.
- **Model access**: in the Bedrock console (Model access), enable the models listed in `config.yaml` (Claude Haiku 4.5, Amazon Nova 2 Lite).
- **IAM permissions** on your SAML role: `bedrock:InvokeModelWithResponseStream` on the foundation models and their inference profile ARNs.
- Verify the configured model IDs exist for your account:

```powershell
aws bedrock list-inference-profiles --region us-east-1 --profile sand --query "inferenceProfileSummaries[].inferenceProfileId"
```

### 6. Run

```powershell
python main.py --list-models      # confirms config loads (no AWS call)
python main.py                    # default model from config.yaml
python main.py --model nova       # start with a specific model
```

### Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `ProfileNotFound` | Profile in `config.yaml` doesn't match the one saml2aws wrote; align the names or pass `--profile`. |
| `ExpiredTokenException` / `UnrecognizedClientException` | Credentials expired; run `saml2aws login` again and restart. |
| `AccessDeniedException` | Model access not enabled, or the role lacks the Bedrock invoke permission. |
| `ResourceNotFoundException` / `ValidationException` on the model | Wrong model ID or region; check with the `list-inference-profiles` command above. |
| `SSL: CERTIFICATE_VERIFY_FAILED ... self signed certificate in certificate chain` | Corporate Zscaler proxy re-signs HTTPS. `config.yaml` sets `aws.ca_bundle: certs/ca-bundle.pem` (public CAs + Sanofi Root/Technical CA). For the AWS CLI, set `$env:AWS_CA_BUNDLE = "D:\Projects\imazeemahmad\ai-agent-aws-traditional-approach\certs\ca-bundle.pem"`. Never disable SSL verification. |
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run. |
| Activation script blocked | See the execution policy note in step 2. |

Options: `--model nova`, `--session <id>` (resume), `--profile <name>`, `--config <file>`, `--list-models`, `--list-sessions`. Run `python main.py -h` for help.

## Commands
`/models`, `/model <name>`, `/kb <question>`, `/kb price <question>`, `/history`, `/new`, `/sessions`, `/help`, `/quit`

## Knowledge base (grounded answers)
With `knowledge_base.enabled: true` the assistant answers only from an Amazon Bedrock Knowledge Base, cites sources,
and refuses (without calling the LLM) when nothing relevant is found. Setup, IAM and tuning steps:
[docs/knowledge-base-setup.md](docs/knowledge-base-setup.md). `/kb <question>` shows retrieved passages and scores.

<!-- improvement 2 ranking step -->
## Relevance reranking
Retrieved passages are reranked (Cohere Rerank 3.5 via the Bedrock Rerank API, same region) before `min_score` is
applied, because vector similarity scores don't separate relevant from irrelevant text. This makes the refusal gate and
the direct price routing work. Choose each KB's `min_score` with `python main.py --calibrate` (labelled questions in
`eval/relevance_set.yaml`). Needs model access to Cohere Rerank 3.5 and IAM `bedrock:Rerank` + `bedrock:InvokeModel`.
Disable with `rerank.enabled: false`. Standard vector and managed Knowledge Bases are both supported (auto-detected).

## Recommended design (target architecture, not yet built)
**Status:** today the main agent decides for itself whether to call the price agent (tool choice). In sessions and in
the evaluation baseline this is unreliable: a compound question such as "stages of commissioning *and* the price of X"
often gets only the exposure half. The recommended fix is to take that decision away from the answering model and make
the flow explicit:

```
question
   |
   v
 follow-up rewrite            (existing)  turn "and its price?" into a standalone query
   |
   v
 ROUTER                       one small LLM call returns: exposure | price | both | out_of_scope
   |
   +-- out_of_scope ---------------------------------> fixed refusal, no further LLM call
   +-- exposure -----> retrieve + rerank (exposure KB) --------------+
   +-- price --------> retrieve + rerank (price KB) ------------------+--> ONE synthesis step
   +-- both ---------> both of the above, run in parallel -----------+     (answers only from the
                                                                          retrieved passages, cites
                                                                          [1] exposure / [P1] price)
```

Design rules:
- **The router decides, the answering model does not.** No "answer only from sources" prompt competing with a tool.
- **Relevance is real.** Both retrievals are reranked and gated by calibrated `min_score` (see *Relevance reranking*),
  so an empty retrieval means "nothing in the KB", not noise.
- **Retrieved passages are presented as system-retrieved context**, never inside the user's message, so the model does
  not say "the sources you provided".
- **One synthesis call** merges the exposure and price evidence, keeps `[1]` and `[P1]` citations distinct, and never
  states a figure that is not in a passage.
- **Keep the audit trail:** route chosen, passages used, sub-agent questions and answers go to `sessions/<id>.json`.
- **Prerequisite outside the code:** one record per product in the price KB (name and price together). No
  orchestration change fixes mixed-up prices from badly chunked source data.

**Framework choice.** Build this first in plain Python behind a small interface (about 100-150 lines on top of the
existing code). Frameworks do not fix the routing problem on their own: Strands "agents as tools" and LangChain agents
use the same model-picks-the-tool pattern that is failing. Adopt one only when you need what it adds:
- **LangGraph**, for an explicit graph with parallel branches, session persistence and human review.
- **Strands Agents on Bedrock AgentCore**, for production deployment, identity and observability (use its graph
  pattern, not agents-as-tools, so routing stays explicit).
- Skip LangChain alone, and skip Bedrock Agents multi-agent collaboration for now (limits model choice).

Whatever is chosen, it must beat the baseline on the evaluation set below.

**Baseline of the current design** (tool choice, `--eval-repeat 3`, 2026-10-19, saved in `eval/results/`):

| Category | Pass | What it shows |
|----------|------|---------------|
| exposure | 100% | exposure answers are fine |
| price | 80% | `price-01` always fails: the agent speculates that two listings are "different versions" (data and prompt issue) |
| **both** | **0%** | **the price agent was never called in 12 of 12 runs** (1 LLM call, 1 retrieval each): the main agent answers the exposure half and says the rest is missing |
| out of scope | 100% | the reranked relevance gate refuses correctly, with no LLM call |
| follow-up | 100% | |
| **overall** | **78%** | route 83%, answer 78%, grounded 100%; avg 3.9 s, 0.9 LLM calls |

The failure is systematic, not flaky: it is a design problem (tool choice), so the router should move "both" from 0% to
near 100% without regressing the other rows.

## Evaluation
Architecture is decided by score, not opinion. `eval/cases.yaml` holds 23 labelled cases: exposure (5), price (5),
both (4), out of scope (6, including "in the art domain but not covered" and a prompt-injection attempt) and follow-ups (3).
Expected answers come from the KB source text. Every check is deterministic (no LLM judge):

| Check | Passes when |
|-------|-------------|
| route | the agents that supplied sources match the expected route (`exposure`, `price`, `both`, `none`) |
| answer | every required fact is present, no forbidden text is present, and out-of-scope cases refuse |
| grounded | every amount, percentage and k/M figure in the answer appears in a retrieved passage or in the question |

```powershell
python main.py --eval --eval-label baseline --eval-repeat 3     # score the current design, 3 runs per case
python main.py --eval --eval-filter both                        # only the "both" cases
python main.py --eval-compare eval/results/A.json eval/results/B.json   # what got fixed / regressed
```

Use `--eval-repeat 3` or more: LLM answers vary, and flaky cases (shown as `FLAKY`) are exactly the routing
unreliability being fixed. Each run reports pass rate, route/answer/grounded accuracy, average latency, average LLM
calls and retrievals (cost), and writes `eval/results/<time>-<label>.json`. To score a new architecture, add a class
with `ask(question, history) -> Result` to `TARGETS` in `src/evaluate.py` and run with `--eval-target <name>`; the
cases and checks do not change. Keep the set honest: add a case for every real failure seen in a session, and update
the expectations when the KB content changes. Results are saved per run, so re-run the baseline after any KB re-sync.

## Testing the agents
Full step-by-step guide (setup, what each test protects, reading the results, adding cases, troubleshooting):
[tests/TESTING.md](tests/TESTING.md). In short, one script, `tests/scripts/run_tests.ps1`, runs everything (PowerShell, uses the project's `.venv`):

| Command | What it does | Needs AWS login |
|---------|--------------|-----------------|
| `.\tests\scripts\run_tests.ps1` | **offline**: 27 unit tests with fake KBs and a fake model (routing, tool loop, labels, reranking, relevance gate, evaluation scoring). About 5 s | no |
| `.\tests\scripts\run_tests.ps1 live` | asks the **real** agents one question per route and shows the routing: who answered, sources per agent, sub-agent calls, timing | yes |
| `.\tests\scripts\run_tests.ps1 live "How much is Fazkurooni?" "Who won the AFL?"` | the same with your own questions (add `-Model nova` to try another model) | yes |
| `.\tests\scripts\run_tests.ps1 eval -Label router-v1 -Repeat 3` | **scored** run of the 23 evaluation cases, saved to `eval/results/` (`-Filter both` for one category) | yes |
| `.\tests\scripts\run_tests.ps1 all` | offline, then live, then eval; skips the live steps if offline fails | yes |

Log in first for the live modes: `saml2aws login --profile sand`. The exit code is non-zero if any step fails.
Run the offline suite after every code change and the eval after every change to prompts, KB content or routing.
The same tests run without the script: `python -m unittest discover -s tests -t . -v`. Tests live in `tests/`
(`test_routing.py`, `test_retrieval.py`, `test_evaluate.py`; `helpers.py` holds the fakes; `live_smoke.py` is the live mode).

## Price sub-agent
A specialist agent with its own price Knowledge Base, called by the main assistant as a tool (`ask_price_agent`).
Pure price questions are routed to it directly. Design, data modelling, AWS/IAM steps and tests:
[docs/price-agent-setup.md](docs/price-agent-setup.md). Set `price_agent.enabled: false` until the price KB exists.
Every part of a reply is labelled with the agent (and model) that produced it: `[main agent | haiku]`,
`[price agent | haiku]`, or `[system]` for the no-match refusal. Hand-offs are shown as `->` lines, and the Sources list
tags each source with its agent. `sessions/<id>.json` records `agent` and the sub-agent `calls` for every answer.
Code layout: `main.py` (entry point), `src/chat.py` (CLI and sessions), `src/assistant.py` (supervisor + price agent),
`src/kb.py` (retrieval), `src/rerank.py` (reranking), `src/calibrate.py` (min_score calibration), `src/evaluate.py` (scoring).

## Config
`config.yaml` holds all settings: AWS region and profile, default model, history length, session folder, the models (ID, max tokens, temperature) and the system prompt.
Add or swap a model by editing that file (PyYAML is required). Each model needs access enabled in the Bedrock console
(Model access) for the region, and your SAML role needs `bedrock:InvokeModelWithResponseStream` and
`bedrock:InvokeModel` (follow-up rewrite, price agent) on the model and its cross-region inference profile, plus
`bedrock:Retrieve` on each knowledge base.

## Sessions
Each run creates a session id. Messages (plus which model produced each reply) are saved to `sessions/<id>.json`.
Switching models keeps the shared history, so the new model sees prior context.
