"""Approved text intersection, connected-group splits, and exact target windows."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .state import digest, fresh_directory, write_json

ROOT = Path("E:/AI/MOTE-quantum")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def approved_content_hash(text):
    # Exact definition in predecessor audit_release.py:8 and pack_stage.py:87.
    return hashlib.sha256(" ".join(text.split()).encode("utf8")).hexdigest()


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while key != root:
            following = self.parent[key]
            self.parent[key] = root
            key = following
        return root

    def union(self, keys):
        roots = [self.find(k) for k in keys]
        target = min(roots)
        for root in roots:
            self.parent[root] = target


def group_split(group):
    value = int.from_bytes(hashlib.sha256(("qwen-joint-v1-split:" + group).encode()).digest()[:8], "big") % 10000
    return "internal_dev" if value < 250 else "reserved_test" if value < 500 else "train"


def prepare_release(output, model_path, queue=ROOT / "queue-20261030"):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Data releases must live in the authorized E: data root")
    inventory = read_json(queue / "data-inventory.json")
    source = inventory["preferred_candidate"]
    manifest = read_json(source["release_manifest"])
    approval = read_json(source["approval_evidence"])
    if (not approval["approved"] or digest(source["release_manifest"]) != approval["manifest_sha256"]
            or digest(source["approved_document_allowlist"]) != manifest["documents"]["sha256"]):
        raise ValueError("Approved source release hash or approval mismatch")
    # Includes worst-case derived files and concurrent installer temporary reservation.
    # Count the whole runtime tree conservatively, including bytecode/test caches.
    budget_roots = [ROOT / "data", ROOT / "runtime"]
    actual = sum(p.stat().st_size for root in budget_roots if root.exists() for p in root.rglob("*") if p.is_file())
    reserve, installer_reserve = 4_000_000_000, 6_000_000_000
    if actual + reserve + installer_reserve > 20_000_000_000:
        raise RuntimeError("Data/cache actual plus reservations would exceed 20GB")
    fresh_directory(output)
    write_json(output / "STORAGE_RESERVATION.json", {"actual_before_bytes": actual, "data_reserved_bytes": reserve,
        "installer_reserved_bytes": installer_reserve, "limit_bytes": 20_000_000_000, "state": "reserved"})
    accepted = {}
    components = Components()
    for row in rows(source["approved_document_allowlist"]):
        if row["language"] != "en":
            continue
        key = row["id"] + ":" + row["content_sha256"]
        if key in accepted:
            accepted[key]["tokens"] += row["tokens"]
            continue
        accepted[key] = row
        keys = ["id:" + row["id"], "content:" + row["content_sha256"]]
        for field in ("split_group", "near_cluster_id", "normalized_sha256"):
            if row.get(field):
                keys.append(field + ":" + row[field])
        components.union(keys)
    # Closed text can predate the final release's cross-source family merges.
    # Union both sets of group links before choosing any split; approval remains
    # the final allowlist's ID + normalized-content hash, never the old group ID.
    available, audited_shards = set(), {}
    for shard in sorted(Path(source["text_root"]).glob(source["accepted_glob"])):
        audited_shards[str(shard)] = digest(shard)
        for row in rows(shard):
            key = row["id"] + ":" + row["content_sha256"]
            if key not in accepted:
                continue
            approved = accepted[key]
            if approved_content_hash(row["text"]) != row["content_sha256"]:
                raise ValueError(f"Approved normalized text mismatch: {key}")
            for field in ("source", "language", "license", "url", "raw_locator"):
                if row.get(field) != approved.get(field):
                    raise ValueError(f"Approved provenance mismatch: {key}, {field}")
            raw_links = ["id:" + row["id"]]
            for field in ("split_group", "near_cluster_id"):
                if row.get(field):
                    raw_links.append(field + ":" + row[field])
                approved["closed_" + field] = row.get(field)
            components.union(raw_links)
            available.add(key)
        if len(audited_shards) % 100 == 0:
            write_json(output / "progress.json", {"phase": "audit_text_and_union_groups_before_tokenization",
                "shards": len(audited_shards), "approved_available_documents": len(available), "time": time.time()})
    groups = defaultdict(list)
    for key, row in accepted.items():
        if key not in available:
            continue
        group = components.find("id:" + row["id"])
        row["qwen_group"] = group
        row["qwen_split"] = group_split(group)
        groups[group].append(key)
    estimates = Counter()
    requested = {"train": 80_000_000, "internal_dev": 3_000_000, "reserved_test": 3_000_000}
    selected = {}
    for group in sorted(groups, key=lambda x: hashlib.sha256(("qwen-joint-v1-select:" + x).encode()).digest()):
        split = group_split(group)
        if estimates[split] >= requested[split]:
            continue
        for key in groups[group]:
            selected[key] = accepted[key]
            estimates[split] += accepted[key]["tokens"]
    if any(estimates[k] < v for k, v in requested.items()):
        raise RuntimeError("Insufficient approved English corpus")
    write_json(output / "document_group_selection.json", {"schema": 1, "grouping": "Transitive union of original ID, content/normalized SHA256, split_group and near_cluster_id",
        "split_salt": "qwen-joint-v1-split:", "split_buckets": {"internal_dev": [0, 249], "reserved_test": [250, 499], "train": [500, 999]},
        "selection_salt": "qwen-joint-v1-select:", "approved_english_unique_records": len(accepted),
        "components": len(groups), "approved_available_documents": len(available),
        "group_version_rule": "Transitive union of final approved release links and corresponding closed-text links before splitting",
        "selected_documents": len(selected), "old_token_estimates_only": dict(estimates)})
    del accepted, groups, components, available
    from transformers import AutoTokenizer
    model_manifest = read_json(queue / "model-manifest.json")
    if Path(model_path).resolve() != Path(model_manifest["path"]).resolve():
        raise ValueError("Model path differs from the frozen user model")
    for entry in model_manifest["files"]:
        if entry["name"] != "model.safetensors" and digest(Path(model_path) / entry["name"]) != entry["sha256"]:
            raise ValueError(f"Model/tokenizer source changed: {entry['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Local tokenizer has no EOS")
    write_json(output / "tokenizer_manifest.json", {"local_model_manifest": model_manifest, "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "add_special_tokens": False,
        "chat_template_applied": False, "packing": "Document text followed by one EOS; fixed 1024 input windows, explicit next targets, ignore document-start targets"})
    goals = {"train": 41_943_040, "internal_dev": 1_048_576, "reserved_test": 1_048_576}
    counts, documents = Counter(), Counter()
    handles = {s: (output / (s + ".tokens.u32")).open("xb") for s in goals}
    starts = {s: [] for s in goals}
    selected_hashes, seen_text, licenses, source_files = set(), set(), {}, []
    record_path = output / "documents.jsonl"
    try:
        with record_path.open("x", encoding="utf8") as records:
            for shard in sorted(Path(source["text_root"]).glob(source["accepted_glob"])):
                shard_hash = digest(shard)
                if shard_hash != audited_shards[str(shard)]:
                    raise ValueError(f"Text changed after group audit: {shard}")
                source_files.append({"path": str(shard), "sha256": shard_hash, "bytes": shard.stat().st_size})
                for row in rows(shard):
                    key = row["id"] + ":" + row["content_sha256"]
                    if key not in selected:
                        continue
                    approved = selected[key]
                    split = approved["qwen_split"]
                    if counts[split] - documents[split] >= goals[split] + 2048 or row["content_sha256"] in seen_text:
                        continue
                    content_hash = approved_content_hash(row["text"])
                    if content_hash != row["content_sha256"] or row["license"] != approved["license"]:
                        raise ValueError(f"Approved text/license mismatch: {row['id']}")
                    for field in ("split_group", "near_cluster_id"):
                        if row.get(field) != approved.get("closed_" + field):
                            raise ValueError(f"Source group changed after split audit: {key}, {field}")
                    tokens = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
                    if len(tokens) < 2:
                        continue
                    if (sum(counts.values()) + len(tokens)) * 4 > reserve // 2:
                        raise RuntimeError("Token output exceeded bounded data reservation")
                    starts[split].append(counts[split])
                    handles[split].write(np.asarray(tokens, dtype="<u4").tobytes())
                    meta = {k: row.get(k) for k in ("id", "title", "source", "source_revision", "language", "license", "url", "raw_locator", "content_sha256", "near_cluster_id", "split_group")}
                    meta.update(qwen_split=split, qwen_group=approved["qwen_group"], start_token=counts[split], tokens=len(tokens), source_file=str(shard))
                    meta["text_utf8_sha256"] = hashlib.sha256(row["text"].encode("utf8")).hexdigest()
                    meta["approved_split_group"] = approved["split_group"]
                    meta["approved_near_cluster_id"] = approved["near_cluster_id"]
                    records.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    licenses[row["source"] + ":" + row["license"]] = {"source": row["source"], "license": row["license"], "source_revision": row.get("source_revision")}
                    counts[split] += len(tokens)
                    documents[split] += 1
                    seen_text.add(content_hash)
                    selected_hashes.add(key)
                if len(source_files) % 100 == 0:
                    write_json(output / "progress.json", {"phase": "tokenizing_approved_text", "shards": len(source_files), "tokens": dict(counts), "documents": dict(documents), "time": time.time()})
                # All selected sources processed so far are hashed; final list is complete for used text.
                if all(counts[s] - documents[s] >= goals[s] + 2048 for s in goals):
                    break
    finally:
        for handle in handles.values():
            handle.close()
    splits = {}
    sampling = {}
    for split, target in goals.items():
        boundaries = np.asarray(starts[split], dtype="<u8")
        boundaries.tofile(output / (split + ".starts.u64"))
        windows = (counts[split] - 1) // 1024
        effective = windows * 1024 - int(((boundaries > 0) & (boundaries <= windows * 1024)).sum())
        if effective < target:
            raise RuntimeError(f"Insufficient actual Qwen supervised targets in {split}: {effective}")
        splits[split] = {"tokens": counts[split], "windows": windows, "valid_targets": effective, "documents": documents[split]}
        for seed in (17, 29, 43):
            order = np.random.Generator(np.random.PCG64(seed)).permutation(windows).astype("<u4")
            filename = f"{split}.seed{seed}.order.u32"
            order.tofile(output / filename)
            sampling[filename] = {"sha256": digest(output / filename), "windows": windows, "generator": "numpy.PCG64", "seed": seed}
    write_json(output / "sampling_manifest.json", sampling)
    write_json(output / "source_files.json", source_files)
    write_json(output / "group_audit_source_hashes.json", audited_shards)
    write_json(output / "LICENSES.json", list(licenses.values()))
    write_json(output / "document_split_hashes.json", {"documents_sha256": digest(record_path), "selection_sha256": digest(output / "document_group_selection.json"),
        "record_count": sum(documents.values()), "exact_text_duplicates_removed": True,
        "near_group_rule": "Union of approved near_cluster_id/split_group and original identity/content hashes before tokenization; inherits approximate near-duplicate coverage of approved audit"})
    files = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file() and p.name not in ("progress.json", "STORAGE_RESERVATION.json")}
    write_json(output / "DATA_RELEASE.json", {"schema": 1, "status": "frozen", "model_path": str(model_path),
        "source_allowlist_sha256": manifest["documents"]["sha256"], "source_approval_sha256": digest(source["approval_evidence"]),
        "sequence_length": 1024, "language": "en", "splits": splits, "files": files,
        "reserved_test_policy": "No model evaluation until matrix/checkpoint/analysis lock", "created_epoch": time.time()})
    write_json(output / "STORAGE_RESERVATION.json", {"state": "complete", "actual_release_bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
        "data_reserved_bytes": 0, "limit_bytes": 20_000_000_000})
    return splits


class TargetWindows:
    def __init__(self, release, split, seed, cursor=None):
        self.release, self.split = Path(release), split
        self.tokens = np.memmap(self.release / (split + ".tokens.u32"), mode="r", dtype="<u4")
        self.starts = np.fromfile(self.release / (split + ".starts.u64"), dtype="<u8")
        self.order = np.fromfile(self.release / f"{split}.seed{seed}.order.u32", dtype="<u4")
        self.position, self.offset = (0, 0) if cursor is None else (cursor["position"], cursor["offset"])

    def state_dict(self):
        return {"position": self.position, "offset": self.offset}

    def next(self, max_targets):
        if max_targets < 1:
            raise ValueError("Positive target count required")
        if self.position >= len(self.order):
            raise StopIteration("Frozen unique stream exhausted; no implicit epochs")
        window = int(self.order[self.position])
        start = window * 1024
        ids = np.asarray(self.tokens[start:start + 1024], dtype=np.int64).copy()
        labels = np.asarray(self.tokens[start + 1:start + 1025], dtype=np.int64).copy()
        left, right = np.searchsorted(self.starts, [start + 1, start + 1025])
        boundary_positions = self.starts[left:right].astype(np.int64) - start - 1
        labels[boundary_positions] = -100
        labels[:self.offset] = -100
        valid = np.flatnonzero(labels != -100)
        used = valid[:max_targets]
        end = int(used[-1]) + 1 if len(used) else 1024
        labels[end:] = -100
        previous_offset = self.offset
        if len(valid) <= max_targets:
            self.position += 1
            self.offset = 0
        else:
            self.offset = end
        return ids, labels, {"window": window, "from_target": previous_offset, "through_target": end, "valid_targets": len(used)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="D:/AI/Models/Qwen3-0.6B")
    args = parser.parse_args()
    existed_before = Path(args.output).exists()
    try:
        print(json.dumps(prepare_release(args.output, args.model)))
    except BaseException:
        output = Path(args.output)
        if not existed_before and output.is_dir():
            write_json(output / "FAILURE.json", {"time": time.time(), "pid": os.getpid(), "error": traceback.format_exc()})
            write_json(output / "STORAGE_RESERVATION.json", {"state": "failed_preserved", "data_reserved_bytes": 0,
                "actual_bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()), "limit_bytes": 20_000_000_000})
        raise


if __name__ == "__main__":
    main()
