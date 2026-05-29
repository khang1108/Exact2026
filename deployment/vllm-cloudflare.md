# Remote vLLM and Cloudflare Tunnel Runbook

This repository should expose the EXACT FastAPI service, not the raw vLLM
server, for challenge use.

```text
external caller
  -> https://exact-api.example.com
  -> Cloudflare Tunnel
  -> EXACT FastAPI on 127.0.0.1:8080
  -> vLLM OpenAI-compatible server on 127.0.0.1:8000
```

The vLLM server is only model infrastructure. The challenge API logic lives in
`exact.app.main:app`, which routes Type 1 and Type 2 requests and returns
`answer`, `explanation`, `fol`, `cot`, `premises`, and `confidence`.

## 1. Start vLLM on the GPU VM

Use an open-source model with 8B parameters or fewer. The default below matches
the project config.

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm

export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000
export VLLM_API_KEY=exact-local-token

scripts/launch_vllm.sh
```

For a quantized AWQ model, override the model and quantization:

```bash
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export VLLM_SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
export VLLM_QUANTIZATION=awq_marlin
scripts/launch_vllm.sh
```

Smoke test vLLM locally on the VM:

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer exact-local-token"
```

Keep vLLM bound to `127.0.0.1` unless the EXACT API runs on a different host.
If the API is on another host in the same private network, bind vLLM to
`0.0.0.0` and restrict access with the VM firewall or private subnet rules.

## 2. Start the EXACT API

In another shell on the same VM:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt

export EXACT_LLM_BASE_URL=http://127.0.0.1:8000/v1
export EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export EXACT_LLM_API_KEY=exact-local-token
export EXACT_LLM_TIMEOUT_SECONDS=120
export EXACT_API_HOST=127.0.0.1
export EXACT_API_PORT=8080

PYTHONPATH=src uvicorn exact.app.main:app --host 127.0.0.1 --port 8080
```

Local API checks:

```bash
curl -s http://127.0.0.1:8080/health

curl -s http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "t2_smoke",
    "question": "Calculate the current when U = 12 V and R = 6 ohm."
  }'
```

## 3. Publish only the EXACT API with Cloudflare Tunnel

Recommended dashboard setup:

1. In Cloudflare Zero Trust, create a tunnel for the VM.
2. Add a public hostname, for example `exact-api.example.com`.
3. Set the service/origin URL to `http://localhost:8080`.
4. Install and run the generated `cloudflared` connector command on the VM.
5. Protect the hostname with Cloudflare Access or another authentication rule.

For a locally managed tunnel, use a config like:

```yaml
tunnel: exact2026-api
credentials-file: /home/ubuntu/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: exact-api.example.com
    service: http://localhost:8080
  - service: http_status:404
```

Validate and run:

```bash
cloudflared tunnel ingress validate
cloudflared tunnel run exact2026-api
```

Cloudflare Tunnel only needs outbound connectivity from the VM to Cloudflare.
Keep inbound ports closed except SSH or your normal admin path.

## 4. Call from another machine

```bash
curl -s https://exact-api.example.com/health

curl -s https://exact-api.example.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "remote_t1",
    "premises-NL": [
      "If a curriculum is well-structured and has exercises, it enhances student engagement.",
      "The curriculum is well-structured.",
      "The curriculum has exercises."
    ],
    "question": "Does the curriculum enhance student engagement?"
  }'
```

`/predict` returns the official EXACT response body: required `answer` and
`explanation`, plus optional `fol`, `cot`, `premises`, and `confidence`. Use
`/debug/predict` when you need local metadata such as `id`, `task_type`,
`question_type`, `unit`, and `error`.

## Operational notes

- Do not tunnel vLLM directly unless you specifically need a public OpenAI-style
  model endpoint. For the challenge, tunnel `/predict` from the EXACT API.
- Set `EXACT_LLM_MODEL` equal to `VLLM_SERVED_MODEL_NAME`.
- Use `EXACT_LLM_API_KEY=EMPTY` only when vLLM is started without `VLLM_API_KEY`.
- Type 1 fails clearly without a JSON LLM client. Type 2 can fall back to some
  deterministic formula logic, but the intended PoT pipeline also needs the LLM.
- The current API has no native authentication middleware. Put Cloudflare Access
  or a gateway rule in front of the tunnel before sharing the URL.
