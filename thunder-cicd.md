# Thunder Compute CI/CD Notes

This note captures the CI/CD direction for deploying the EXACT 2026 project on
Thunder Compute.

## Current Project Shape

The repository currently looks like a Python service and research workspace:

- `pyproject.toml` and `requirements.txt` define Python dependencies.
- `Dockerfile` is present and intends to run a FastAPI/Uvicorn service on port
  `8080`.
- `README.md` describes a FastAPI health check at `GET /health`.
- Notebook baseline `src/exact/baselines/B01_zero_shot.ipynb` uses an 8B-or-
  smaller local LLM path:
  - Hugging Face model ID: `Qwen/Qwen2.5-7B-Instruct`
  - Ollama model ID: `qwen2.5:7b`
- `src/exact/config.py` already exposes environment-driven LLM settings such as
  model name, provider, base URL, API key, timeouts, and mock mode.

## Repo Gaps To Resolve Before CD

Do these before automating production deployment:

1. Confirm and fix the application entrypoint.
   The Dockerfile currently starts `uvicorn app.main:app`, but an `app/main.py`
   entrypoint was not found during inspection. API code was found under
   `src/exact/app/`.
2. Make the deployed API useful enough to smoke test.
   `/health` exists, but `/predict` in `src/exact/app/router.py` is still a
   placeholder.
3. Add tests.
   No `tests/` folder was found during inspection.
4. Add CI workflows.
   No `.github/workflows/` folder was found during inspection.
5. Decide the production model-serving path.
   The notebook supports Ollama and Hugging Face loading. The API service should
   have one clearly documented runtime path before deployment is automated.

## Thunder Compute Facts That Affect The Design

Thunder Compute documentation currently states:

- CLI/script authentication can use the `TNR_API_TOKEN` environment variable.
- Instances can be created through CLI or API.
- Thunder has `prototyping` and `production` instance modes.
- Prototyping mode is intended for R&D, experimentation, fine-tuning, training,
  and small-scale inference.
- Long-running inference services and unattended production jobs should use
  production mode.
- Thunder instances have persistent disk storage; optional ephemeral storage is
  mounted at `/ephemeral` and is suitable for caches, weights, and scratch data
  that can be lost.
- Public HTTP access can be exposed with Thunder port forwarding, including
  `tnr ports forward`.
- File transfer is available through `tnr scp`.
- Snapshots are the mechanism to preserve instance state and later restore it.
- Instance templates include at least `base` and `ollama`.

Relevant docs read during the discussion:

- Thunder Compute CLI quickstart
- Authentication docs
- Creating instances docs
- Prototyping vs Production docs
- Technical specifications docs
- File transfer docs
- Docker guide
- Snapshot/stop workflow docs

## Recommended First Architecture

Use one Thunder Compute production instance for the deployed inference service:

```text
Thunder Compute production instance
  Ollama server
    qwen2.5:7b
  EXACT FastAPI service
    calls the local LLM server
  systemd
    manages API process and restart behavior
```

This is the pragmatic first path because:

- the project already demonstrates `qwen2.5:7b` in its notebook baseline;
- Thunder offers an Ollama-oriented instance template;
- FastAPI and model serving can be restarted and debugged independently;
- normal CI can stay CPU-only and avoid expensive GPU work on every pull
  request.

## Docker Position

The repo already has a Dockerfile, so image-based deployment is possible.
However, Thunder currently documents Docker support as experimental inside
Thunder instances. The docs also say Docker Compose does not work there and GPU
containers require Thunder's supported GPU device flag rather than copying
generic Docker GPU examples.

Recommendation:

1. Start with host deployment using a Python virtual environment plus `systemd`.
2. Keep Docker builds in CI to catch packaging regressions.
3. Move CD to image-based deployment later if the Thunder Docker path proves
   stable for this service.

## CI Design

Run CI on pull requests and pushes to `main`.

### Standard CPU CI

The default workflow should:

1. Check out the repository.
2. Set up Python 3.11 or 3.12.
3. Install dependencies.
4. Run linting, initially `ruff`.
5. Run `pytest`.
6. Build the Docker image.
7. Start the service in mock mode when possible and check `/health`.

### Test Layers

Add focused test layers:

- Unit tests for schemas, config, routing decisions, parsing, and symbolic/math
  logic.
- API tests for `/health`, prediction request validation, errors, and batch
  request behavior when available.
- LLM contract tests with mocked model responses for output parsing and failure
  handling.
- Real GPU/model smoke tests only in a separate workflow.

### What Not To Put In Every PR

Do not run a real 7B model inference job for every PR. It is slower, more
expensive, less deterministic, and couples routine CI to GPU capacity.

## CD Design

### First CD Path: Host Deployment

For a development or staging Thunder instance:

1. Require CI success.
2. Authenticate Thunder automation with a scoped `TNR_API_TOKEN`.
3. Transfer or pull the release onto the target instance.
4. Create or update a Python virtual environment.
5. Install project dependencies.
6. Set runtime environment variables.
7. Restart the FastAPI `systemd` service.
8. Check `/health`.
9. Run one small prediction smoke test after `/predict` is implemented.

### Alternative CD Path: Docker Image

Once validated on Thunder:

1. Build the Docker image in GitHub Actions.
2. Push it to a registry such as GHCR.
3. Pull the image on the Thunder instance.
4. Run the API container.
5. Forward/expose the API port.
6. Run health and prediction smoke tests.

### Ephemeral GPU Validation Path

Use a manually triggered or scheduled workflow when reproducibility matters:

1. Create a Thunder instance.
2. Deploy the exact code revision.
3. Pull or restore the required model assets.
4. Run GPU smoke tests or benchmark jobs.
5. Save needed artifacts.
6. Snapshot or delete the instance according to cost and reuse needs.

## Branch And Environment Strategy

Use this release shape:

- Pull request:
  - lint
  - tests
  - Docker build
  - mock/API health checks
- `main`:
  - same CI
  - optional automatic deploy to dev or staging
- release tag such as `v*`, or a GitHub Environment approval:
  - deploy to production Thunder instance
  - run post-deploy smoke tests

Notebook experiments should not automatically redeploy production just because a
research cell changed.

## Secrets And Environment Variables

Prepare CI/CD secrets carefully. Likely GitHub Actions secrets include:

- `TNR_API_TOKEN`
- Thunder target instance name or ID for each environment
- deploy SSH material if the chosen connection path requires it
- model registry or Hugging Face token only if gated downloads are required
- any challenge submission credentials added later

Runtime EXACT settings should be documented from `src/exact/config.py`. Relevant
settings include:

- `EXACT_LLM_MODEL` or `EXACT_MODEL_ID`
- `EXACT_LLM_BASE_URL`
- `EXACT_LLM_API_KEY` if needed by the chosen backend
- `EXACT_MOCK_LLM` or `MOCK_LLM`
- token/timeouts/retry settings
- API host, port, environment, and log settings

Do not commit secrets or generated token files.

## Rollback And Cost Control

Use a simple rollback path first:

1. Keep the previous git revision or previous container image available.
2. Restart the previous release if the post-deploy smoke test fails.
3. Snapshot before major environment or model-server changes.

For cost control:

- keep regular CI CPU-only;
- use GPU workflows only for meaningful smoke tests, benchmarks, or releases;
- avoid leaving temporary GPU instances alive after validation;
- treat model caches and large scratch files intentionally, using persistent or
  ephemeral storage based on whether they must survive instance replacement.

## Suggested Implementation Roadmap

1. Fix the FastAPI app entrypoint and local start command.
2. Make `/health` and `/predict` suitable for smoke tests.
3. Add a `tests/` suite.
4. Add `.github/workflows/ci.yml`.
5. Create a Thunder production instance for long-running inference.
6. Run Ollama with `qwen2.5:7b`, or document another model server explicitly.
7. Add a deploy script and a `systemd` unit for the API.
8. Add `.github/workflows/deploy-dev.yml`.
9. Add production deployment with approval or release tags.
10. Add a separate GPU/model smoke-test workflow.

## Minimal Deliverables For The First CI/CD Pass

The first concrete submission should include:

- `.github/workflows/ci.yml`
- `.github/workflows/deploy-dev.yml`
- a deployment script for Thunder
- a `systemd` unit for the FastAPI service
- documented Thunder secrets and runtime environment variables
- tests for the API health path and request validation
- a post-deploy smoke check

That is enough to show a real CI/CD path without overbuilding infrastructure
before the deployed API and model-server contract are finalized.
