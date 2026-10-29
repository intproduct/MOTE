from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import IMAGE_TASKS, data_key, digest

IMAGE_REPOS = {
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "cifar10": "uoft-cs/cifar10",
}
TEXT_FIELDS = {
    "sst2": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}


def cache_path(cfg):
    return Path(cfg["cache_dir"]) / data_key(cfg)


def stratified_indices(labels, count, seed):
    """Seeded proportional subsample without replacement (no sklearn dependency)."""
    labels = np.asarray(labels)
    count = min(int(count), len(labels))
    if count < 1:
        raise ValueError("empty split")
    g = np.random.default_rng(seed)
    values, sizes = np.unique(labels, return_counts=True)
    raw = sizes * count / len(labels)
    take = np.floor(raw).astype(int)
    order = np.argsort(-(raw - take), kind="stable")
    take[order[: count - int(take.sum())]] += 1
    out = np.concatenate(
        [
            g.permutation(np.flatnonzero(labels == label))[:n]
            for label, n in zip(values, take)
        ]
    )
    return g.permutation(out).tolist()


def split_train(labels, cfg):
    dev_n = min(cfg["dev_limit"], max(1, int(len(labels) * cfg["dev_fraction"])))
    dev = stratified_indices(labels, dev_n, cfg["data_seed"])
    pool = np.setdiff1d(np.arange(len(labels)), dev)
    train_local = stratified_indices(
        np.asarray(labels)[pool], cfg["train_limit"], cfg["data_seed"] + 1
    )
    return pool[train_local].tolist(), dev


def _image_tensor(dataset):
    field = "img" if "img" in dataset.column_names else "image"
    array = np.stack([np.array(x, copy=True) for x in dataset[field]])
    return {
        "features": (
            torch.from_numpy(array).float().reshape(len(array), -1) / 255 - 0.5
        )
        / 0.5,
        "labels": torch.tensor(dataset["label"], dtype=torch.long),
    }


def _text_tensor(dataset, tokenizer, task, length):
    fields = TEXT_FIELDS[task]
    encoded = tokenizer(
        *[list(dataset[f]) for f in fields],
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
    )
    return dict(encoded, labels=torch.tensor(dataset["label"], dtype=torch.long))


def prepare(cfg, stage="all", tasks=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    path = cache_path(cfg)
    path.mkdir(parents=True, exist_ok=True)
    selected = [
        (s, t)
        for s in ("mlp", "bert")
        if stage in {"all", s}
        for t in cfg[s]["tasks"]
        if tasks is None or t in tasks
    ]
    tokenizer = None
    if any(s == "bert" for s, t in selected):
        info_path = path / "bert.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            model_path = snapshot_download(
                cfg["bert_model"],
                revision=info["commit"],
                allow_patterns=[
                    "*.json",
                    "vocab.txt",
                    "*.safetensors",
                    "pytorch_model.bin",
                ],
            )
        else:
            commit = (
                HfApi().model_info(cfg["bert_model"], revision=cfg["bert_revision"]).sha
            )
            model_path = snapshot_download(
                cfg["bert_model"],
                revision=commit,
                allow_patterns=[
                    "*.json",
                    "vocab.txt",
                    "*.safetensors",
                    "pytorch_model.bin",
                ],
            )
            info_path.write_text(
                json.dumps({"repo": cfg["bert_model"], "commit": commit}, indent=2)
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
    for s, task in selected:
        destination = path / f"{task}.pt"
        if destination.exists() and (path / f"{task}.json").exists():
            print(f"prepared cache exists: {destination}", flush=True)
            continue
        repo = IMAGE_REPOS[task] if task in IMAGE_TASKS else "nyu-mll/glue"
        revision = HfApi().dataset_info(repo).sha
        ds = load_dataset(
            repo, **({} if task in IMAGE_TASKS else {"name": task}), revision=revision
        )
        train_indices, dev_indices = split_train(ds["train"]["label"], cfg)
        final_name = "test" if task in IMAGE_TASKS else "validation"
        final_indices = stratified_indices(
            ds[final_name]["label"], cfg["test_limit"], cfg["data_seed"] + 2
        )
        selected_ds = {
            "train": ds["train"].select(train_indices),
            "dev": ds["train"].select(dev_indices),
            "test": ds[final_name].select(final_indices),
        }
        tensors = {
            k: _image_tensor(v)
            if task in IMAGE_TASKS
            else _text_tensor(v, tokenizer, task, cfg["max_length"])
            for k, v in selected_ds.items()
        }
        indices = {"train": train_indices, "dev": dev_indices, "test": final_indices}
        meta = {
            "task": task,
            "repo": repo,
            "revision": revision,
            "data_key": data_key(cfg),
            "final_source_split": final_name,
            "classes": len(set(ds["train"]["label"])),
            "input_dim": tensors["train"].get("features", torch.empty(0, 0)).shape[-1],
            "indices": indices,
            "indices_sha256": digest(indices),
            "fingerprints": {k: v._fingerprint for k, v in selected_ds.items()},
            "sizes": {k: len(v["labels"]) for k, v in tensors.items()},
        }
        torch.save(tensors, destination.with_suffix(".tmp"))
        destination.with_suffix(".tmp").replace(destination)
        (path / f"{task}.json").write_text(json.dumps(meta, indent=2))
        print(f"prepared {task}: {meta['sizes']}", flush=True)


def load_data(cfg, task):
    path = cache_path(cfg)
    meta = json.loads((path / f"{task}.json").read_text())
    if meta["data_key"] != data_key(cfg):
        raise ValueError("prepared data config mismatch")
    data = torch.load(path / f"{task}.pt", map_location="cpu", weights_only=True)
    meta["tensor_file_sha256"] = hashlib.sha256(
        (path / f"{task}.pt").read_bytes()
    ).hexdigest()
    return data, meta


def bert_path(cfg):
    from huggingface_hub import snapshot_download

    info = json.loads((cache_path(cfg) / "bert.json").read_text())
    return snapshot_download(
        info["repo"], revision=info["commit"], local_files_only=True
    ), info


class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors["labels"])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensors.items()}
