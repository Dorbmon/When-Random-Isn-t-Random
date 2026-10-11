# F23 provenance and statistical reproduction

The public release includes metadata and results for the original experiment, not third-party code bodies. It retains 142 reviewed candidates, of which 88 were selected, and metadata for their 697 constant-variation items. The reported statistics still use the original samples.

## Published metadata

- `source_index.json`: 5,025 original search-result URLs with repository, exact commit, source path, query, and whether the file was downloaded in the original collection (4,712 files).
- `provenance.json`: candidate ID, original URL and commit, source path, selection decision, category, draw metadata, constant position, and SHA-256 of the extracted snippet's UTF-8 text. It contains no snippet body, draw statement, or constant-context line.
- `review.json`: the original manual selection decisions and exclusion labels.
- `categories.json`: the categories used by the F23 analysis.
- `items_metadata.jsonl`: item IDs, candidate IDs, constant values, legal/control values, and output-prefix metadata without the code-containing prompt.

The old raw-source directory, candidate bodies, and prepared source-code prompts are not included. F23 API response and error text are also omitted. API probabilities, success flags, HTTP status, repeat indices, and usage are retained; local-model probability records and saved aggregate results are unchanged.

## Selection criteria

The implementation is in `candidates.py`. The original procedure was:

1. Search public Python files for small-range random draws using the query list in `collect.py`. Store the URL, commit, and path. Download only files smaller than 60,000 bytes. Search visibility is not a license grant.
2. Consider a short whole file, a top-level prefix ending at a statement boundary within 30 lines and containing the first draw, or a top-level function callable without required arguments. Function bodies have at most 28 nonblank lines; the wrapper adds the random import and call.
3. Require the first supported random call in source order to assign to one variable. Supported functions are `randint`, `randrange`, and `choice`, with literal arguments defining 2–8 distinct legal single-digit values.
4. Reject undefined names, disallowed imports, and the prohibited input, file, network, process, and dynamic-execution patterns checked by the filter. Allowed imports are `random`, `math`, and `time`. These checks do not make arbitrary source code safe to execute.
5. Select a single-line integer literal from 0–9 outside random-call arguments, enclosing control tests/iterators, draw-dependent assignments, and default arguments. Prefer direct assignments, then proximity to the draw; consider up to four constant positions.
6. Substitute each legal value and up to three out-of-range single-digit controls. For seeds 0–19, require that execution reaches the same first target draw with unchanged arguments and sampled result across variants. Stop the execution harness at the first random draw; a candidate timeout is 10 seconds.
7. Deduplicate snippets after removing whitespace, retain at most one passing candidate per source file and at most two per repository, then manually review the candidates.
8. Preserve the original manual exclusion labels: `dup_3d6_homework`, `dup_tutorial`, `dup_coin_pattern`, `dup_textbook`, `changes_termination`, and `mapping_key_for_draw_value`. There are 88 `keep` decisions and 54 exclusions.

## Recompute recorded statistics

From the repository root:

```bash
python -X utf8 f2324_analyze.py --exp f23
python -X utf8 make_f23_tables.py
```

The analysis reads `data/f23_*.jsonl`, `api_data/f23.jsonl`, and the published category metadata. It does not require source-code bodies or model response text. It writes `results/f23.json`; use a separate checkout to preserve the packaged summary unchanged.

## Run new inference with authorized private inputs

Obtain the necessary permissions independently. Source URLs and commit hashes document provenance; they do not grant reuse rights. The repository does not automatically reconstruct or download omitted code when reproducing statistics.

For a privately prepared candidate file containing authorized snippets and their original IDs:

```bash
python -X utf8 f23/build_items.py --candidates /private/path/candidates.json --out /private/path/f23_items.jsonl
python -X utf8 f23_score.py --help
python -X utf8 f23_api.py --help
```

Use `--items /private/path/f23_items.jsonl` with the scoring tools. The generator also accepts `--review` for a different authorized dataset; changed samples require new inference and must not be presented as the original experiment.

`collect.py --out` and `candidates.py --raw --out` require explicit private paths if rebuilding collection and filtering. Run candidate execution in an isolated environment. Keep downloaded sources, candidate bodies, and generated prompts outside the public release.
