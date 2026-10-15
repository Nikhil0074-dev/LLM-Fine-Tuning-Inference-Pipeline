# LLM Fine-Tuning & Inference Pipeline

Upload a dataset, fine-tune a model, test it against a baseline, deploy it only if it passes, and serve it
through a secured API. It runs fully on your laptop for free. The same code can run on AWS
(S3, Step Functions, SageMaker).

## 1. Read this first: what is proven and what is not

| Part | Status |
|---|---|
| API, auth, tenants, datasets, jobs, gate, registry, deploy, rollback, cache, rate limit, metrics | **Tested.** 43 automated tests pass. Also run live over HTTP. |
| S3 storage, Redis cache | **Tested** with simulated S3 (moto) and simulated Redis (fakeredis). Not tested on real AWS or a real Redis server. |
| Step Functions flow, Lambda callback, SageMaker entry script | **Logic tested** with stand-ins (payloads, callbacks, security checks). Never run on real AWS. |
| LoRA fine-tuning (`app/ml/lora.py`) | **Not executed.** No GPU and no Hugging Face access where this was built. It compiles and follows the standard Transformers + PEFT pattern, but expect to fix small version issues on first run. |
| Terraform | **Not applied or validated.** Run `terraform validate` and `plan` first. |

Nobody can honestly promise "error-free" for cloud code that has never run on the cloud.
Run in this order: local path, then LoRA on a tiny model, then AWS.

## 2. What makes this design different

- **Two model backends, one pipeline.** `simple` is a small no-GPU model (TF-IDF nearest neighbour). It lets you test the whole
  pipeline for free, in CI, in seconds. `lora` does real fine-tuning. Same API, same gate, same registry.
- **A gate with three outcomes.** `approved`, `rejected`, or `needs_review`. A model is compared with the *baseline*
  (not just a fixed score), and checked for errors, empty answers and slow answers. If the validation set is under 20 rows,
  the score is too noisy to trust, so a human must approve.
- **"Training finished" is not "model approved".** A job can succeed and still produce a rejected model. A rejected model
  cannot be deployed or overridden by hand. It must be retrained.
- **One owner of the decision.** The cloud only reports metrics. The API applies the gate. No gate logic is copied into Lambda.
- **Cost guard before the GPU starts.** Each job gets a cost estimate. If it would pass the monthly budget, it is refused.
  There is also a cap on active jobs per tenant, and a required `Idempotency-Key`, so a retry never starts a second paid job.
- **Cache keys that never go stale.** The key holds tenant, immutable model-version id, prompt, all generation settings and
  a prompt-template version. A new model can never read old answers. A rollback safely reuses its own old answers. No invalidation code is needed.
- **Safe deploy.** Deploy runs a smoke test first. If it fails, the active model does not change. Rollback is one call.
- **Privacy by default.** Prompts and answers are never written to the database. Datasets with emails, keys or card numbers are blocked
  and never stored. Detected values are never echoed back in reports.

## 3. Architecture

```
Client ──X-API-Key──> FastAPI (app/)
                        │  datasets · jobs · registry · inference · metrics
        ┌───────────────┼────────────────────────┐
     SQLite          Storage                    Cache
  (jobs, models,    local files or S3     memory or Redis
   deployments,     (datasets, models)    (answers, rate limits)
   audit, logs)
                        │ start job
          ┌─────────────┴──────────────┐
   LocalOrchestrator             StepFunctionsOrchestrator
   (thread, any backend)         MarkTraining ─> SageMaker job ─> Register
                                  (Lambda)     (train + evaluate)   (Lambda -> API /internal)
```

Training and serving are separate. Lambda only passes messages. It never trains or serves a model.

## 4. Quick start (local, free)

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests                      # 43 tests

export LLMP_API_KEYS="admin-key-0123456789ab:admin:demo:alice;member-key-0123456789:member:demo:bob"
uvicorn app.main:app_factory --factory         # docs at http://127.0.0.1:8000/docs
python scripts/demo.py --key admin-key-0123456789ab
```

Key format: `key:role:tenant:name`, keys at least 16 characters. `admin` can upload, train, review, deploy and read metrics.
`member` can call inference and read.

Docker: copy `.env.example` to `.env`, then run `docker compose up --build` (API + Redis).

### Your own data
JSONL (`{"prompt": "...", "response": "..."}` per line) or CSV with `prompt,response` columns.
For classification, `response` is the label, and you set `"task_type": "classification"`.

## 5. API

| Method | Path | Role | Purpose |
|---|---|---|---|
| POST | `/api/v1/datasets?name=&format=` | admin | Upload the file as request body. Validates and splits. |
| GET | `/api/v1/datasets`, `/{id}` | any | Report: counts, duplicates, PII flags, label spread |
| POST | `/api/v1/training-jobs` | admin | Needs an `Idempotency-Key` header (8–100 chars) |
| GET | `/api/v1/training-jobs`, `/{id}` | any | Status and progress |
| POST | `/api/v1/training-jobs/{id}/cancel` | admin | Cooperative stop (local) or stop execution (AWS) |
| GET | `/api/v1/models`, `/{id}` | any | Versions, metrics vs baseline, reasons |
| POST | `/api/v1/models/{id}/review` | admin | Approve or reject a `needs_review` model |
| POST | `/api/v1/models/{id}/deploy` | admin | Smoke test, then atomic switch |
| GET | `/api/v1/deployments` | any | History (newest first) |
| POST | `/api/v1/deployments/{name}/rollback` | admin | Restore previous version |
| POST | `/api/v1/inference` | any | Generate. Returns cache flag, cold-start flag, token usage |
| GET | `/api/v1/metrics` | admin | p50/p95 cached vs uncached, hit rate, tokens, spend |
| GET | `/api/v1/health` | none | Database, storage, cache checks |

Training request example:
```json
{"dataset_id": "ds_...", "model_name": "support-bot", "backend": "lora",
 "task_type": "generation",
 "hyperparameters": {"epochs": 3, "learning_rate": 0.0002},
 "gate": {"min_score": 0.5, "min_improvement": 0.05, "max_p95_latency_ms": 5000}}
```
All numbers are range-limited. Unknown fields are rejected. Backends and base models come from an allow-list.

## 6. Run real LoRA fine-tuning (do this before AWS)

```bash
pip install -r requirements-training.txt
export LLMP_ALLOWED_BACKENDS=simple,lora
```
Create a job with `"backend": "lora"`. The default base model is `HuggingFaceTB/SmolLM2-135M-Instruct`
(small, slow on CPU). Add others with `LLMP_ALLOWED_BASE_MODELS`, and check each model's license.
For a first run, use 30–100 rows and 1 epoch. On CPU, raise the gate's `max_p95_latency_ms`, because CPU generation is slow.
GPU cost estimates count against `LLMP_MONTHLY_BUDGET_USD` (default 50).

## 7. Deploy to AWS

1. Build `Dockerfile.training` and push it to ECR.
2. In `infrastructure/terraform`, set `bucket_name`, `training_image`, `api_url`, `internal_api_key`. Run `validate`, `plan`, then `apply`.
3. Run the API image (`Dockerfile`) on ECS/Fargate or App Runner behind HTTPS. Attach the `api_runtime_policy_arn` output to its task role.
   Set: `LLMP_STORAGE_BACKEND=s3`, `LLMP_S3_BUCKET`, `LLMP_S3_PREFIX`, `LLMP_ORCHESTRATOR=stepfunctions`,
   `LLMP_STATE_MACHINE_ARN`, `LLMP_INTERNAL_API_KEY`, `LLMP_REDIS_URL`, `LLMP_API_KEYS`.
4. Set an **AWS Budgets alert before the first job**. The built-in estimate is a guard rail, not a bill.
5. Check your SageMaker quota for the GPU instance. New accounts often have zero.

Limits of the shipped cloud setup:
- **SQLite is single-node.** Run one API container, or move to PostgreSQL before scaling out.
- **The model is loaded inside the API container.** That is simple and fast, but it is not serverless and needs
  enough memory for the model. For GPU serving, add a model-endpoint class behind the `Predictor` interface in `app/ml/`.
- Uploads go through the API (5 MB default). Large files need presigned S3 uploads.

## 8. Files

```
app/        main, api, security, datasets, jobs, registry, inference, evaluation, cost, cache, storage, orchestrator
app/ml/     base interface, simple backend, lora backend
training/   sagemaker_entry.py  (trains, then evaluates base vs tuned, writes metrics.json)
lambda_functions/callback.py    workflows/training_state_machine.asl.json
infrastructure/terraform/main.tf    Dockerfile    Dockerfile.training    docker-compose.yml
tests/      43 tests        scripts/  demo.py, make_demo_dataset.py
```

## 9. Honest limits and blind spots

- **Demo scores are inflated.** The sample dataset is synthetic, and validation prompts are close rewrites of training prompts.
  A score of 1.0 there proves the pipeline works, not that a model is good. Real data needs a properly held-out set.
- **Word-overlap scores do not measure answer quality** for free text. Start with classification tasks, or add human review.
  The gate is a safety net, not proof of a good model.
- **Fine-tuning changes style, not facts.** For changing documents (policies, prices), use retrieval (RAG).
- **PII detection is pattern-based.** It finds emails, card numbers (Luhn-checked), AWS keys, private keys and common tokens.
  It does not find names, addresses or phone numbers.
- **Rate limits are per API key, per minute.** They are shared across servers only if Redis is configured.
  If Redis is down, limits stop applying but serving continues.
- **A timeout does not kill a running generation.** The caller gets a 504, but the thread finishes in the background.
- **The cost estimate uses an assumed speed (1500 tokens/s) and an assumed price.** Set `LLMP_GPU_HOURLY_USD` from the current AWS price list.
