<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->


## Repository modification rules

- Treat all existing branches as protected history.
- Never modify, rewrite, rebase, reset, force-push, delete, or otherwise alter any pre-existing branch unless the user explicitly requests it.
- Work only on the branch that the user explicitly selects for the current task.
- If the user later creates or selects another branch, switch to that branch and apply all new changes there. Previous branches must remain unchanged.
- Never merge changes into another branch unless explicitly requested.
- Never create a new branch unless explicitly requested by the user.

## Before changing code

- Read this `AGENTS.md` first.
- Use Graft before broad source-code inspection to understand the relevant subsystem, implementation, callers, and blast radius.
- Prefer:
  - `graft map` for repository orientation.
  - `graft ask "<question>" --source` for understanding implementation.
  - `graft callers <symbol>` for call relationships.
  - `graft skeleton <file>` for API/file structure.
  - `graft grep "<literal>"` when exhaustive matching is required.
- After using Graft, inspect only the source ranges/files needed for the task.
- Before editing, identify which files must change and which behavior must remain unchanged.

## Code-change policy

- Make the smallest change that correctly implements the requested behavior.
- Do not refactor unrelated code.
- Do not rename, move, reformat, or clean up unrelated files.
- Do not change existing experiment behavior, baseline semantics, random-number behavior, dataset handling, training configuration, evaluation logic, or defaults unless the requested task explicitly requires it.
- Preserve backward compatibility whenever practical.
- New experimental behavior should preferably be opt-in and should not silently change the original VarDrop execution path.
- Keep research/diagnostic functionality separate from the original baseline path when possible.
- Do not remove existing diagnostics, logging, measurements, or experimental functionality merely because they are not part of the current main method.
- Existing experimental code may be isolated or disabled from the main execution path when appropriate, but must not be deleted without explicit approval.

## Reproducibility and regression safety

- Changes must not silently alter existing results.
- When modifying shared training, sampling, data-loading, inference, or evaluation code, add or run appropriate regression/parity checks.
- When an optimized implementation is intended to be mathematically or semantically equivalent to an existing implementation, explicitly verify equivalence.
- Prefer checkpoint-based or lightweight regression tests before requesting expensive retraining.
- Do not launch large or expensive experiment sweeps unless explicitly requested.
- Do not introduce new random seeds, repeated runs, or additional training runs unless explicitly requested.

## Experiment logs and reports

- Raw experiment reports are stored under `report/`.
- Treat existing files under `report/` as immutable experimental evidence.
- Never delete, overwrite, truncate, rename, or modify an existing report unless explicitly requested.
- New runs must create new report files rather than replacing previous reports.
- Preserve detailed logs needed for later paper analysis, plotting, statistical analysis, timing analysis, and reproducibility.
- Do not reduce logging simply to make output cleaner.
- For performance benchmarking, measurement-specific logging may be isolated from timed regions so logging overhead does not corrupt timing results.
- Graft is for source-code understanding; raw `.txt` experiment reports should be searched/read directly with normal file tools or `rg`.

## Validation after changes

After making code changes:

1. Review the complete diff.
2. Confirm that only intended files and behaviors changed.
3. Run relevant lightweight tests or sanity checks.
4. Run regression/parity checks when shared behavior was touched.
5. Report:
   - files changed,
   - behavior changed,
   - behavior intentionally preserved,
   - tests performed,
   - any remaining risks or assumptions.

Do not proceed to expensive experiments until the code-change and regression checks pass.
