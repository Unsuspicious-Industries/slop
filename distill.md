Distillation workflow
=====================

This document explains the distillation tooling and how to run a teacher‑based
collection workflow. The repository includes small adapters for several teacher
backends (Grok, OpenAI, DALL·E, and OpenRouter). The tooling reads credentials
from a local `.env` file when present.

Goals
-----
- Collect teacher-generated images for a set of prompts (or a dataset of labels)
- Store a manifest and sample files for later training/evaluation
- Keep the collection step safe (rate-limited / concurrent) and reproducible

Key locations
-------------
- Distill tooling: `distill/` (`distill/collect.py`, `distill/config.py`, `distill/prompts.py`)
- Teacher adapters: `distill/teachers/` (`grok.py`, `openai.py`, `dalle.py`, `openrouter.py`)
- Example prompts: `distill/prompts.csv`
- Historical dataset embeddings & labels: `data/historical/embeddings/labels.npy`
- Notebook (collection smoke test / workflow): `notebooks/distill_workflow.ipynb`

Prerequisites
-------------
- Python dependencies listed in `requirements.txt` (install into a venv).
- A teacher API key in `.env` for the backend you plan to use. The distillation
  tooling respects existing environment variables; `.env` is only loaded when
  present and does not overwrite existing vars.

Supported .env keys
-------------------
- `XAI_API_KEY` — used by `GrokTeacher` (xAI / Grok)
- `OPENAI_API_KEY` — used by `OpenAITeacher` (OpenAI images)
- `OPENROUTER_API_KEY` — used by `OpenRouterTeacher` (OpenRouter-compatible image API)
- Optional OpenRouter overrides: `OPENROUTER_URL`, `OPENROUTER_MODEL`

Creating a `.env` file (example)
-------------------------------
Create a file named `.env` at the repository root (or in the working directory
where you run the distill commands) with lines like:

OPENROUTER_API_KEY=sk_xxxxxx
# Optional overrides
OPENROUTER_URL=https://api.openrouter.ai/v1
OPENROUTER_MODEL=gpt-image-1

Running a smoke test from the notebook
-------------------------------------
1. Open `notebooks/distill_workflow.ipynb` in JupyterLab or VS Code.
2. Ensure `.env` contains your teacher key (see above).
3. Run the first cells: they load prompts derived from
   `data/historical/embeddings/labels.npy` and instantiate the configured
   teacher (default: `openrouter`).
4. The smoke cell generates one sample and writes `distill/historical_samples/sample_0.png`.

Using the CLI to collect a dataset
---------------------------------
The primary collection entrypoint is `distill.collect.collect(cfg)` which the
repo exposes via the CLI wrapper `python -m distill.collect`.

Example: collect using the historical prompts and OpenRouter teacher

    python -m distill.collect --run-id historical_run --dataset-name historical \
        --teacher openrouter --model-preset sd15 --samples-per-prompt 2 \
        --max-concurrent 4

Key flags
- `--teacher`: one of `grok`, `openai`, `dalle`, `openrouter`.
- `--model-preset`: `sd15`, `sd21`, `sdxl` — sets default student model and training defaults.
- `--samples-per-prompt`: number of teacher images per prompt.
- `--max-concurrent`: concurrency for async collection (API rate & cost control).
- `--partial-supervision`: enable partial image streaming when supported by teacher.

Where outputs go
----------------
The collector writes into the directory configured in `DistillConfig.output_dir`.
By default the output path is `distill/data/<dataset_name>/<run_id>/` with
subfolders `samples/`, `logs/` and `artifacts/`.

Historical dataset integration
------------------------------
The repository includes a simple loader that turns entries in
`data/historical/embeddings/labels.npy` into prompts (see
`distill/prompts.load_historical_prompts`). Each label is converted to a
prompt of the form `Photograph of {label}`. You can pass `max_count` to limit
the number of prompts used for collection.

Choosing a teacher
------------------
- `grok`: uses `XAI_API_KEY` and the xAI client.
- `openai`: uses `OPENAI_API_KEY` and the OpenAI async client; supports
  streaming partial images when enabled.
- `dalle`: placeholder adapter for DALL·E-style backends (implementations vary).
- `openrouter`: lightweight adapter for OpenRouter-compatible endpoints; reads
  `OPENROUTER_API_KEY` from `.env` and tries several response shapes to extract
  base64 image data.

Notes on cost, rate-limits and ethics
------------------------------------
- Teacher APIs incur usage costs — keep `samples_per_prompt` and concurrency low
  while experimenting.
- Historical collections can include sensitive material. Follow the project's
  safety and ethics guidance when building datasets and publishing results.

Troubleshooting
---------------
- If no image is returned, inspect the teacher adapter's error message — the
  OpenRouter adapter will raise a helpful error showing the HTTP response.
- If the collector hangs or you see repeated SSH banners when using remote
  providers: see the main README and the client `clearmem`/restart helpers.
- If `labels.npy` contains bytes, the prompt loader tries to decode them as
  UTF‑8; if labels are non-textual identifiers you may want to map ids to
  human-readable captions first.

Next steps (ideas)
------------------
- Add a small prompt normalization step for historical labels to create richer
  prompts (e.g. add time/place modifiers).
- Add a multi-teacher comparison mode that queries several teachers for the
  same prompt and stores outputs side-by-side for qualitative analysis.
- Add quota-aware rate limiting (token bucket) per teacher adapter.

Questions or changes
--------------------
If you want the notebook to run the full `collect()` flow automatically, or
prefer a single-file CLI wrapper that sets `.env` keys and runs a job, tell me
which option and I will add it.
