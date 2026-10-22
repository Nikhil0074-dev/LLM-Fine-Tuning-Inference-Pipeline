from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("llmp.orchestrator")


class LocalOrchestrator:
    """Runs jobs on a thread pool inside the API process. Good for development and tests."""

    def __init__(self, workers: int):
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="job")
        self.ctx = None  # set by build_context

    def start(self, job: dict, dataset: dict) -> str:
        from .jobs import run_local_job

        self.pool.submit(run_local_job, self.ctx, job["id"])
        return f"local:{job['id']}"

    def cancel(self, job: dict) -> bool:
        return False  # cooperative: the running job checks the cancel flag


class StepFunctionsOrchestrator:
    """Starts the AWS Step Functions state machine (see workflows/ and infrastructure/)."""

    def __init__(self, settings, client=None):
        if not settings.state_machine_arn:
            raise ValueError("LLMP_STATE_MACHINE_ARN is required for the stepfunctions orchestrator")
        if client is None:
            import boto3

            client = boto3.client("stepfunctions", region_name=settings.aws_region)
        self.client = client
        self.settings = settings
        self.ctx = None

    def start(self, job: dict, dataset: dict) -> str:
        s = self.settings
        config = json.loads(job["config_json"])
        hp = {k: str(v) for k, v in config["hyperparameters"].items()}
        hp.update(base_model=config["base_model"], task_type=config["task_type"],
                  metrics_s3_uri=self.ctx.storage.uri(f"{job['tenant']}/jobs/{job['id']}/metrics.json"))
        payload = {
            "job_id": job["id"],
            "tenant": job["tenant"],
            "training_job_name": f"llmp-{job['id'].replace('_', '-')}",
            "instance_type": s.sagemaker_instance_type,
            "hyperparameters": hp,
            "train_prefix_uri": self.ctx.storage.uri(dataset["train_key"]),
            "val_prefix_uri": self.ctx.storage.uri(dataset["val_key"]),
            "output_uri": self.ctx.storage.uri(f"{job['tenant']}/jobs/{job['id']}/output") + "/",
            "metrics_s3_uri": hp["metrics_s3_uri"],
        }
        resp = self.client.start_execution(
            stateMachineArn=s.state_machine_arn, name=f"job-{job['id'].replace('_', '-')}",
            input=json.dumps(payload))
        return resp["executionArn"]

    def cancel(self, job: dict) -> bool:
        if not job.get("execution_ref"):
            return False
        self.client.stop_execution(executionArn=job["execution_ref"], cause="Cancelled by user")
        return True
