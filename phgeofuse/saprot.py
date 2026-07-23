from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from .cache import artifact_path, atomic_torch_save, sha256_text, valid_torch_cache
from .config import get, load_config, path
from .io import ProteinRecord, read_manifest, write_manifest


def saprot_text(sequence: str, three_di: str) -> str:
    if len(sequence) != len(three_di):
        raise ValueError("SaProt amino-acid and 3Di sequences must have equal lengths")
    supported = frozenset("ACDEFGHIKLMNPQRSTVWY")
    return "".join(
        (amino_acid if amino_acid in supported else "#") + state.lower()
        for amino_acid, state in zip(sequence.upper(), three_di, strict=True)
    )


def embedding_key(sequence: str, three_di: str, config: dict[str, Any]) -> str:
    return sha256_text(
        "\0".join(
            [
                str(get(config, "model.saprot_model", "westlake-repl/SaProt_650M_AF2")),
                str(get(config, "model.saprot_revision", "main")),
                sequence,
                three_di,
            ]
        )
    )


def embedding_cache_path(
    root: str | Path, sequence: str, three_di: str, config: dict[str, Any]
) -> Path:
    return artifact_path(root, embedding_key(sequence, three_di, config), ".pt")


class SaProtEncoder:
    def __init__(self, config: dict[str, Any], device: torch.device, trainable: bool = False):
        self.config = config
        self.device = device
        self.model_name = str(get(config, "model.saprot_model", "westlake-repl/SaProt_650M_AF2"))
        revision = get(config, "model.saprot_revision")
        offline = bool(get(config, "runtime.offline", False))
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from transformers.utils.hub import cached_file

        options: dict[str, Any] = {"local_files_only": offline}
        if revision:
            options["revision"] = revision
        model_source = self.model_name
        if offline:
            cached_config = cached_file(
                self.model_name, "config.json", revision=revision, local_files_only=True
            )
            model_source = str(Path(cached_config).parent)
            options.pop("revision", None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **options)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        full_model = AutoModelForMaskedLM.from_pretrained(
            model_source, torch_dtype=dtype, low_cpu_mem_usage=True,
            use_safetensors=False, **options
        )
        self.model = full_model.esm if hasattr(full_model, "esm") else full_model.base_model
        del full_model
        self.model.to(device)
        self.model.train(trainable)
        for parameter in self.model.parameters():
            parameter.requires_grad_(trainable)
        if trainable and bool(get(config, "model.gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable()

    def encode(self, texts: list[str], lengths: list[int]) -> list[torch.Tensor]:
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = tokenized.pop("special_tokens_mask").to(self.device).bool()
        inputs = {key: value.to(self.device) for key, value in tokenized.items()}
        context = torch.enable_grad() if self.model.training else torch.inference_mode()
        with context:
            hidden = self.model(**inputs).last_hidden_state
        residue_mask = inputs["attention_mask"].bool() & ~special
        rows = []
        for index, length in enumerate(lengths):
            row = hidden[index][residue_mask[index]]
            if row.shape[0] != length:
                raise ValueError(
                    f"SaProt token/residue mismatch: {row.shape[0]} tokens for {length} residues"
                )
            rows.append(row)
        return rows


def encode_records(
    records: Iterable[ProteinRecord],
    config: dict[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    records = [record for record in records if record.status == "ready"]
    local = records[rank::world_size]
    pending = [record for record in local if not valid_torch_cache(
        record.embedding_path,
        {
            "sequence_sha256": record.sequence_sha256,
            "embedding_key": embedding_key(record.sequence, record.three_di, config),
        },
    )]
    if not pending:
        return 0
    encoder = SaProtEncoder(config, device, trainable=False)
    batch_size = int(get(config, "encoding.batch_size", 1))
    written = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        texts = [saprot_text(record.sequence, record.three_di) for record in batch]
        rows = encoder.encode(texts, [len(record.sequence) for record in batch])
        for record, row in zip(batch, rows, strict=True):
            payload = {
                "embedding": row.detach().to(dtype=torch.float16, device="cpu").contiguous(),
                "metadata": {
                    "sequence_sha256": record.sequence_sha256,
                    "embedding_key": embedding_key(record.sequence, record.three_di, config),
                    "model": encoder.model_name,
                    "length": len(record.sequence),
                },
            }
            atomic_torch_save(record.embedding_path, payload)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SaProt residue embeddings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--distributed-backend", default="nccl")
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("runtime", {})["offline"] = bool(get(config, "runtime.offline", False))
    manifest = Path(args.manifest).resolve() if args.manifest else path(config, "paths.manifest")
    records = read_manifest(manifest)

    from utils.distributed import initialize_distributed

    context = initialize_distributed(args.distributed_backend)
    try:
        written = encode_records(
            records, config, context.device, rank=context.rank, world_size=context.world_size
        )
        count = torch.tensor(float(written), device=context.device)
        context.all_reduce(count)
        context.barrier()
        if context.is_main:
            write_manifest(manifest, records)
            print(f"Cached {int(count.item())} SaProt embeddings under {path(config, 'paths.embeddings')}")
    finally:
        context.close()


if __name__ == "__main__":
    main()
