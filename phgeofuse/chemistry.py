from __future__ import annotations

import torch


IONIZABLE_RESIDUES = ("D", "E", "C", "Y", "H", "K", "R")
IONIZABLE_TO_INDEX = {residue: index for index, residue in enumerate(IONIZABLE_RESIDUES)}
STANDARD_PKA = {"D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1, "H": 6.0, "K": 10.5, "R": 12.5}
ACIDIC_INDICES = frozenset(range(4))
BASIC_INDICES = frozenset(range(4, 7))


def residue_ionization(sequence: str, predicted_pka: dict[int, float] | None = None):
    predicted_pka = predicted_pka or {}
    types, pkas, missing = [], [], []
    for position, residue in enumerate(sequence):
        type_index = IONIZABLE_TO_INDEX.get(residue, -1)
        types.append(type_index)
        if type_index < 0:
            pkas.append(7.0)
            missing.append(0.0)
        elif position in predicted_pka:
            pkas.append(float(predicted_pka[position]))
            missing.append(0.0)
        else:
            pkas.append(STANDARD_PKA[residue])
            missing.append(1.0)
    return (
        torch.tensor(types, dtype=torch.long),
        torch.tensor(pkas, dtype=torch.float32),
        torch.tensor(missing, dtype=torch.float32),
    )


def henderson_hasselbalch_charge(
    ionizable_type: torch.Tensor,
    pka: torch.Tensor,
    ph: torch.Tensor,
) -> torch.Tensor:
    """Return residue charge with shape ``[..., residues, pH points]``."""

    expanded_pka = pka.unsqueeze(-1)
    expanded_ph = ph.reshape(*([1] * pka.dim()), -1)
    acidic = (ionizable_type >= 0) & (ionizable_type <= 3)
    basic = ionizable_type >= 4
    acid_charge = -1.0 / (1.0 + torch.pow(10.0, expanded_pka - expanded_ph))
    base_charge = 1.0 / (1.0 + torch.pow(10.0, expanded_ph - expanded_pka))
    return torch.where(
        acidic.unsqueeze(-1),
        acid_charge,
        torch.where(basic.unsqueeze(-1), base_charge, torch.zeros_like(acid_charge)),
    )
