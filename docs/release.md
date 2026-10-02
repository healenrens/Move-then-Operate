# Publication boundary

Publish source, English documentation, configuration templates and offline tests. Keep datasets, videos, weights, labels, raw API replies, error sidecars, manifests with deployment paths, simulator results, logs, virtual environments and W&B directories on deployment storage. `.gitignore` prevents adding untracked artifacts; it does not remove already tracked files.

Inject Ark configuration using `ARK_API_KEY`, `ARK_MODEL_ID` and optional `ARK_BASE_URL`. Inject W&B credentials using `WANDB_API_KEY`. The legacy annotation `--api_key` argument remains compatible, but environment injection avoids placing a key in command-line examples. No deployment-specific model endpoint is a default. API exceptions are redacted before annotation retry output, feedback and error sidecars.

`python scripts/check_release.py --root <source-export>` checks an exported publication tree. For a development checkout, pass a NUL-separated list from `git ls-files -z` using `--file-list`. Include newly staged source when preparing that list. The checker reports filenames, line numbers and categories without matched values. It detects non-English text, credential candidates, private deployment paths/addresses/endpoints, private keys, runtime artifacts and symlinks. It performs no content hashing.

A working-tree check reads current file contents. The Git index controls the next commit, and Git history contains earlier commits. A clean current tree and ignore rules do not clean history. Maintainers own any necessary credential rotation or history cleanup; this release workflow does not rewrite history.
