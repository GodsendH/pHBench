from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .cache import atomic_json, atomic_text, sha256_file
from .config import get


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    number: int
    insertion: str
    name: str
    amino_acid: str
    ca: tuple[float, float, float]
    plddt: float


def sequence_matches(observed_sequence: str, target_sequence: str) -> bool:
    """Match equal-length sequences while treating target X residues as unknown."""
    return len(observed_sequence) == len(target_sequence) and all(
        target == "X" or observed == target
        for observed, target in zip(observed_sequence, target_sequence, strict=True)
    )


def _sequence_match_metadata(
    target_sequence: str, residues: list[Residue]
) -> dict[str, Any]:
    observed_sequence = "".join(residue.amino_acid for residue in residues)
    ambiguous_positions = [
        index + 1 for index, residue in enumerate(target_sequence) if residue == "X"
    ]
    return {
        "observed_sequence": observed_sequence,
        "sequence_match_mode": "x_wildcard" if ambiguous_positions else "exact",
        "ambiguous_positions": ambiguous_positions,
    }


def parse_pdb_chains(path: str | Path) -> dict[str, list[Residue]]:
    chains: dict[str, list[Residue]] = {}
    seen: set[tuple[str, int, str]] = set()
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            name = line[17:20].strip().upper()
            amino_acid = AA3_TO_1.get(name)
            if amino_acid is None:
                continue
            chain = line[21:22].strip() or "A"
            number = int(line[22:26])
            insertion = line[26:27].strip()
            key = (chain, number, insertion)
            if key in seen:
                continue
            seen.add(key)
            residue = Residue(
                chain=chain,
                number=number,
                insertion=insertion,
                name=name,
                amino_acid=amino_acid,
                ca=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
                plddt=float(line[60:66]),
            )
            chains.setdefault(chain, []).append(residue)
    if not chains:
        raise ValueError(f"no C-alpha atoms found in structure: {path}")
    return chains


def select_chain(path: str | Path, target_sequence: str) -> tuple[str, list[Residue]]:
    chains = parse_pdb_chains(path)
    matches = [
        (chain, residues)
        for chain, residues in chains.items()
        if sequence_matches(
            "".join(residue.amino_acid for residue in residues), target_sequence
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("multiple structure chains match the input sequence")
    if len(chains) == 1:
        _, residues = next(iter(chains.items()))
        observed = "".join(residue.amino_acid for residue in residues)
        raise ValueError(
            f"structure sequence mismatch: expected {len(target_sequence)} residues, "
            f"observed {len(observed)}"
        )
    raise ValueError("no unique structure chain matches the input sequence")


def download_alphafold_structure(
    protein_id: str,
    sequence: str,
    destination: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import requests

    api_template = str(
        get(config, "structure.alphafold_api", "https://alphafold.ebi.ac.uk/api/prediction/{protein_id}")
    )
    timeout = int(get(config, "structure.download_timeout", 60))
    attempts = int(get(config, "structure.download_attempts", 3))
    api_url = api_template.format(protein_id=protein_id)
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            pdb_url = payload.get("pdbUrl") or payload.get("pdb_url")
            if not pdb_url:
                raise RuntimeError(f"AlphaFold DB response has no PDB URL: {api_url}")
            structure_response = requests.get(pdb_url, timeout=timeout)
            structure_response.raise_for_status()
            atomic_text(destination, structure_response.text)
            chain, residues = select_chain(destination, sequence)
            return {
                "source": "alphafold_db",
                "url": pdb_url,
                "chain": chain,
                "mean_plddt": sum(item.plddt for item in residues) / len(residues),
                "structure_sha256": sha256_file(destination),
                **_sequence_match_metadata(sequence, residues),
            }
        except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            Path(destination).unlink(missing_ok=True)
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"AlphaFold DB lookup failed for {protein_id}: {error}")


def predict_esmfold_structure(
    protein_id: str,
    sequence: str,
    destination: str | Path,
    config: dict[str, Any],
    offline: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold fallback requires a CUDA GPU")
    repo = str(get(config, "structure.esmfold_model", "facebook/esmfold_v1"))
    revision = get(config, "structure.esmfold_revision")
    kwargs = {"local_files_only": offline}
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.utils.hub import cached_file
    if revision:
        kwargs["revision"] = revision
    model_source = repo
    if offline:
        cached_config = cached_file(repo, "config.json", revision=revision, local_files_only=True)
        model_source = str(Path(cached_config).parent)
        kwargs.pop("revision", None)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    model = EsmForProteinFolding.from_pretrained(
        model_source, low_cpu_mem_usage=True, use_safetensors=False, **kwargs
    )
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(int(get(config, "structure.esmfold_chunk_size", 64)))
    model = model.cuda().eval()
    tokens = tokenizer([sequence], add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    try:
        with torch.inference_mode():
            output = model(tokens)
        pdb = model.output_to_pdb(output)[0]
        atomic_text(destination, pdb)
        chain, residues = select_chain(destination, sequence)
        return {
            "source": "esmfold",
            "model": repo,
            "chain": chain,
            "mean_plddt": sum(item.plddt for item in residues) / len(residues),
            "structure_sha256": sha256_file(destination),
            **_sequence_match_metadata(sequence, residues),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def acquire_structure(
    protein_id: str,
    sequence: str,
    destination: str | Path,
    config: dict[str, Any],
    offline: bool,
) -> dict[str, Any]:
    destination = Path(destination)
    provenance_path = destination.with_suffix(destination.suffix + ".json")
    if destination.is_file() and provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            _, residues = select_chain(destination, sequence)
            if provenance.get("structure_sha256") == sha256_file(destination):
                return {**provenance, "reused": True, "mean_plddt": sum(r.plddt for r in residues) / len(residues)}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if offline:
        raise FileNotFoundError(f"offline structure cache is missing or invalid: {destination}")
    try:
        provenance = download_alphafold_structure(protein_id, sequence, destination, config)
    except RuntimeError as alphafold_error:
        provenance = predict_esmfold_structure(protein_id, sequence, destination, config, offline=False)
        provenance["alphafold_error"] = str(alphafold_error)
    provenance.update({"protein_id": protein_id, "sequence": sequence, "reused": False})
    atomic_json(provenance_path, provenance)
    return provenance


def foldseek_three_di(
    structure_path: str | Path,
    sequence: str,
    plddt: torch.Tensor,
    config: dict[str, Any],
) -> tuple[str, str]:
    binary = shutil.which(str(get(config, "structure.foldseek_binary", "foldseek")))
    if binary is None:
        raise FileNotFoundError("Foldseek executable is not available on PATH")
    with tempfile.TemporaryDirectory(prefix="phgeofuse-foldseek-") as directory:
        output = Path(directory) / "descriptor.tsv"
        command = [
            binary, "structureto3didescriptor", "-v", "0", "--threads", "1",
            "--chain-name-mode", "1", str(structure_path), str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Foldseek failed: {result.stderr.strip()}")
        rows: list[tuple[str, str, str]] = []
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                columns = line.rstrip("\n").split("\t")
                if len(columns) >= 3:
                    rows.append((columns[0], columns[1].upper(), columns[2].lower()))
    matches = [row for row in rows if sequence_matches(row[1], sequence)]
    if len(matches) != 1:
        raise ValueError("Foldseek did not return one chain matching the input sequence")
    _, observed, three_di = matches[0]
    if len(three_di) != len(sequence):
        raise ValueError("Foldseek 3Di and amino-acid sequence lengths differ")
    threshold = float(get(config, "structure.plddt_mask_threshold", 70.0))
    masked = "".join("#" if float(confidence) < threshold else state for state, confidence in zip(three_di, plddt, strict=True))
    return observed, masked


def run_propka(
    structure_path: str | Path,
    residues: list[Residue],
    config: dict[str, Any],
) -> tuple[dict[int, float], str | None]:
    binary = shutil.which(str(get(config, "structure.propka_binary", "propka3")))
    if binary is None:
        return {}, "PROPKA executable is unavailable"
    with tempfile.TemporaryDirectory(prefix="phgeofuse-propka-") as directory:
        source = Path(directory) / Path(structure_path).name
        shutil.copy2(structure_path, source)
        result = subprocess.run([binary, str(source), "--quiet"], cwd=directory, capture_output=True, text=True)
        outputs = list(Path(directory).glob("*.pka"))
        if result.returncode != 0 or not outputs:
            return {}, (result.stderr or "PROPKA produced no output").strip()
        text = outputs[0].read_text(encoding="utf-8", errors="replace")
    index_by_key = {(res.name, res.number, res.chain): index for index, res in enumerate(residues)}
    values: dict[int, float] = {}
    in_summary = False
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        columns = line.split()
        if len(columns) < 4 or columns[0] not in AA3_TO_1:
            continue
        try:
            number, chain, value = int(columns[1]), columns[2], float(columns[3])
        except ValueError:
            continue
        index = index_by_key.get((columns[0], number, chain))
        if index is not None:
            values[index] = value
    return values, None
