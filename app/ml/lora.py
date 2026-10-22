"""LoRA fine-tuning with Hugging Face Transformers + PEFT.

Heavy libraries (torch, transformers, peft) are imported inside functions so that the API and the
test-suite work without them. The same code runs locally and inside the SageMaker container.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import threading
import zipfile
from typing import Callable

from .base import Backend, GenResult, Predictor, TrainingCancelled

TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"
META_FILE = "pipeline_meta.json"

DEFAULT_HP = {"epochs": 3, "learning_rate": 2e-4, "batch_size": 4, "max_seq_len": 256,
              "lora_r": 8, "lora_alpha": 16, "seed": 42}


def _safe_extract_zip(data: bytes, dest: str) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = os.path.abspath(dest)
        for member in zf.namelist():
            target = os.path.abspath(os.path.join(dest, member))
            if not target.startswith(root + os.sep) and target != root:
                raise ValueError("unsafe path in artifact")
        zf.extractall(dest)


def _safe_extract_tar(data: bytes, dest: str) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        root = os.path.abspath(dest)
        for member in tf.getmembers():
            target = os.path.abspath(os.path.join(dest, member.name))
            if not (target == root or target.startswith(root + os.sep)):
                raise ValueError("unsafe path in artifact")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("links and devices are not allowed in artifacts")
        try:
            tf.extractall(dest, filter="data")  # Python 3.12+: extra safety on top of the checks above
        except TypeError:
            tf.extractall(dest)


def extract_artifact(data: bytes, dest: str) -> None:
    """Accepts the zip made locally or the model.tar.gz made by SageMaker."""
    if zipfile.is_zipfile(io.BytesIO(data)):
        _safe_extract_zip(data, dest)
    else:
        _safe_extract_tar(data, dest)


def zip_directory(path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(path):
            for name in files:
                full = os.path.join(folder, name)
                zf.write(full, os.path.relpath(full, path))
    return buf.getvalue()


def train_lora(train_rows: list[dict], base_model: str, hp: dict, output_dir: str,
               should_cancel: Callable[[], bool] = lambda: False) -> None:
    """Fine-tune and write the adapter + tokenizer + pipeline_meta.json to output_dir."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    hp = {**DEFAULT_HP, **hp}
    max_len = int(hp["max_seq_len"])
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = get_peft_model(model, LoraConfig(
        r=int(hp["lora_r"]), lora_alpha=int(hp["lora_alpha"]), lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules="all-linear"))

    examples = []
    for row in train_rows:
        p_ids = tok(TEMPLATE.format(prompt=row["prompt"]), add_special_tokens=False)["input_ids"]
        r_ids = tok(row["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]  # loss only on the response
        if any(label != -100 for label in labels):
            examples.append({"input_ids": ids, "labels": labels})
    if not examples:
        raise ValueError("no usable training examples (are prompts longer than max_seq_len?)")

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}

    class CancelCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if should_cancel():
                control.should_training_stop = True
            return control

    with tempfile.TemporaryDirectory() as scratch:
        args = TrainingArguments(
            output_dir=scratch, num_train_epochs=float(hp["epochs"]),
            per_device_train_batch_size=int(hp["batch_size"]), learning_rate=float(hp["learning_rate"]),
            seed=int(hp["seed"]), logging_steps=5, save_strategy="no", report_to=[],
            remove_unused_columns=False, fp16=torch.cuda.is_available())
        Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate,
                callbacks=[CancelCallback()]).train()
    if should_cancel():
        raise TrainingCancelled()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    with open(os.path.join(output_dir, META_FILE), "w") as fh:
        json.dump({"base_model": base_model, "hyperparameters": hp}, fh)


class HFPredictor:
    def __init__(self, base_model: str, adapter_dir: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter_dir or base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self._lock = threading.Lock()  # one generation at a time per loaded model

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenResult:
        torch = self._torch
        text = TEMPLATE.format(prompt=prompt)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        kwargs = {"max_new_tokens": int(max_new_tokens), "pad_token_id": self.tok.pad_token_id}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            kwargs.update(do_sample=False)
        with self._lock, torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return GenResult(text=self.tok.decode(new_tokens, skip_special_tokens=True).strip(),
                         input_tokens=int(inputs["input_ids"].shape[1]), output_tokens=int(len(new_tokens)))


class LoraBackend(Backend):
    name = "lora"
    uses_gpu = True

    def train(self, train_rows, config, should_cancel):
        with tempfile.TemporaryDirectory() as out:
            train_lora(train_rows, config["base_model"], config.get("hyperparameters", {}), out, should_cancel)
            return zip_directory(out)

    def load(self, artifact: bytes) -> Predictor:
        with tempfile.TemporaryDirectory() as d:
            extract_artifact(artifact, d)
            with open(os.path.join(d, META_FILE)) as fh:
                meta = json.load(fh)
            return HFPredictor(meta["base_model"], d)  # weights are in memory after this call

    def baseline(self, train_rows, config) -> Predictor:
        return HFPredictor(config["base_model"], None)
