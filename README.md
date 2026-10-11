# When Random Isn't Random

Code, source provenance, experiment metadata, probability tables, and analysis results for the context-repulsion experiments. Third-party F23 source code and prompts containing that code are not distributed.

## Install

Use Python 3.11 or 3.12. Run commands from this directory. The CPU analysis dependencies below were checked with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -X utf8 verify_artifacts.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Use `python -X utf8` for the scripts so that recorded Unicode responses are read consistently across platforms.

## Files

| Path | Contents |
| --- | --- |
| `*.py` | Experiment definitions, model scoring, API collection, and statistical analysis |
| `natural_factorization.py` | Shared chat-template rendering and digit-token enumeration |
| `data/` | Local-model probability tables, interventions, generations, and API-to-grid conversions |
| `api_data/` | API probabilities, usage, request status, and recorded responses except F23 response/error text |
| `f24_items.jsonl` | Prepared synthetic dice experiment inputs |
| `f23/` | Source URLs and commits, candidate/item metadata, selection decisions, categories, and filtering code; see [F23 usage](f23/README.md) |
| `results/` | Saved aggregate statistics and per-cell results, including intermediate analyses |
| `output_snapshots/` | Distinct earlier captures retained alongside the primary output files; directory names are content-hash prefixes |
| `recipes/` | Experiment launch commands, including model-specific separators and selected attention heads |
| `chat_templates/` | Saved chat templates for Gemma, Muse, and Qwen3-32B |
| `artifact_manifest.json` | File sizes, SHA-256 checksums, and JSONL row counts |

The primary directories consolidate the available captures. Exact duplicate files are omitted where they occupy the same logical path. Distinct earlier versions remain in `output_snapshots/`; do not concatenate them with primary files, as they can overlap. Pilot and unsuccessful requests remain available; the analysis loaders apply their validity filters. Provider-issued `response_id` values are cleared for anonymity. F23 response and error text are omitted to avoid redistributing source-code excerpts; its probabilities, request status, usage, sample/repeat indices, and saved statistics are preserved. Blank JSONL files represent captures with zero rows. Some statistical JSON files use Python's `NaN` or `Infinity` representation for undefined or unbounded quantities; Python's `json` reader supports these values.

F23 provenance retains original GitHub URLs and exact commits without distributing the referenced code. Source links do not grant permission to use or redistribute the source. The recorded F23 statistics still describe the original 88 selected snippets; no samples were removed from the statistical analysis. Model names identify the models evaluated. The repository contains recorded outputs, not model weights.

## Analyze existing outputs

These commands use the included outputs and make no model API requests:

```bash
mkdir -p reproduced
python -X utf8 analyze_repulsion.py --local data/Qwen3-8B.jsonl --out reproduced/qwen8.json
python -X utf8 analyze_repulsion.py --logprobs api_data/logprobs.jsonl --lp-specs gpt-6-luna --out reproduced/api_logprobs.json
python -X utf8 analyze_repulsion.py --api api_data/raw.jsonl --api-specs gpt-6-luna --out reproduced/api_samples.json
```

On PowerShell, create the output directory with `New-Item -ItemType Directory -Force reproduced`.

The following scripts reproduce the corresponding saved analyses. They write into `results/`, replacing the matching saved summaries, so use a separate checkout if you want to retain the packaged files unchanged:

```bash
python -X utf8 robust_stats.py
python -X utf8 robust_stats2.py
python -X utf8 robust_stats3.py
python -X utf8 robust_stats4.py
python -X utf8 spurious_assoc.py
python -X utf8 f2324_analyze.py --exp f23
python -X utf8 f2324_analyze.py --exp f24
python -X utf8 f25_analyze.py
```

Additional `analyze_f*.py` scripts cover individual experiments. Scripts using `argparse` expose their arguments with `--help`; compact analysis utilities accept input filenames as positional arguments. `make_f23_tables.py` and `make_spurious_tables.py` format the saved results as tables.

## Run local models

Local-model inference requires a CUDA GPU, sufficient memory for the selected model, and locally downloaded weights and tokenizers:

```bash
python -m pip install -r requirements-gpu.txt
mkdir -p reproduced
python -X utf8 repulsion_score.py --model /path/to/Qwen3-8B --model-name Qwen3-8B --family qwen --x-separator '  \n' --out reproduced/Qwen3-8B.jsonl
```

`--family` selects `qwen`, `gemma`, or `muse`. Gemma and Muse runs use `AutoModelForMultimodalLM`; install a Transformers version supporting the exact checkpoint architecture. Pass `--tokenizer /path/to/tokenizer` when the chat template is supplied separately. To use a saved template, copy the corresponding `chat_templates/*.jinja` file to `chat_template.jinja` inside your downloaded tokenizer directory. The loaders use `local_files_only=True`. GPU dependencies are not pinned because the required build depends on the checkpoint and CUDA installation; the release checks do not establish GPU inference compatibility.

The two spaces before `\n` in the Qwen3-8B command are intentional. Consult the recipes for each model's separator, tokenizer override, and intervention settings. To use a Bash recipe, set `MODEL_DIR` to a directory containing the model subdirectories, `TOKENIZER_DIR` to the tokenizer directory, and optionally `PYTHON` to the interpreter executable:

```bash
export MODEL_DIR=/path/to/models
export TOKENIZER_DIR=/path/to/tokenizers
bash recipes/run_f2526.sh
```

These batch recipes can require two GPUs and write to the primary `data/` directory. Run them in a separate checkout. Individual Python commands give direct control over output paths. The `GEMMA_MODEL` and optional `GEMMA_TOKENIZER` environment variables configure `gemma_mask_diag.py`.

## Collect new API outputs

Set `OPENROUTER_API_KEY` in your environment. The API scripts retain the model identifiers and request settings used in the recorded experiments; availability and pricing may differ when rerunning. New requests incur provider charges.

```bash
python -X utf8 api_repulsion.py --specs gpt-6-luna --samples 2 --budget-usd 1 --out reproduced/api_new.jsonl
python -X utf8 api_logprobs.py --help
python -X utf8 f23_api.py --help
```

Use a fresh output filename for a new experiment. The collectors support resumption; successful observations already in the selected ledger are skipped. Preserve repeat/sample indices when analyzing captures.

## Rebuild inputs

```bash
python -X utf8 natural_factorization.py --cases-only --out reproduced/cases
python -X utf8 f24_dice.py
```

These commands regenerate synthetic inputs. Existing F23 statistics can be recomputed directly from the published probability records. Rerunning F23 inference requires independently authorized source material, which is not included; see [F23 usage and selection criteria](f23/README.md). The collection, candidate filtering, and prompt-building tools require explicit private input/output paths. `f23/candidates.py` executes candidate snippets during filtering and must be used in an isolated environment with appropriate source permissions.
