"""Independently reconstruct the weighted pilot and compare matched inner fits."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from localph.residue_field import ResidueField
from localph.residue_training import predict, choose
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData
from phgeofuse.delta_ref.metrics import metrics, acceptance


def main():
    base = ROOT / "experiments/residue_field_phopt_20260917"
    pilot = base / "rarity_pilot"
    protocol = json.loads((pilot / "protocol.json").read_text())
    for section in ("source_hashes", "baseline_hashes"):
        for name, digest in protocol[section].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError("pilot provenance changed")
    feature = base / "features/tokens.npz"
    if sha256_file(feature) != protocol["features_sha256"]:
        raise ValueError("feature hash differs")
    d = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    train = d.train
    keys, y, folds, groups = d.keys[train], d.labels[train], d.folds[train], d.groups[train]
    fit, query = np.flatnonzero(~np.isin(folds, [0, 1])), np.flatnonzero(folds == 1)
    if protocol["fit_keys"] != keys[fit].tolist() or protocol["query_keys"] != keys[query].tolist() or set(groups[fit]) & set(groups[query]):
        raise ValueError("pilot isolation differs")
    with np.load(feature, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("feature key alignment differs")
        packed = {k: z[k].copy() for k in ("tokens", "ionizable", "offsets")}
    bdir = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42/excluded_0_1"
    with np.load(bdir / "predictions.npz", allow_pickle=False) as z:
        lookup = dict(zip(z["keys"], z["prediction"]))
        baseline = np.array([lookup[k] for k in keys[query]])
    bm = metrics(y[query], baseline, groups[query])
    torch.set_num_threads(2)
    result, rows = {}, []
    for kind in ("direct", "sparse"):
        for name, folder in (("natural", base / "nested_v2/outer0" / kind / "inner1"), ("weighted", pilot / kind)):
            report = json.loads((folder / "complete.json").read_text())
            with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                decisions = {k: z[k].copy() for k in z.files if k != "keys"}
            choice = choose(y[query], baseline, decisions)
            fused = np.clip(baseline + choice["strength"] * (decisions[choice["decision"]] - baseline), 0, 14)
            m = metrics(y[query], fused, groups[query])
            check = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
            model = ResidueField(kind=kind)
            model.load_state_dict(check["state_dict"])
            reloaded = predict(model, packed, query[:32], np.zeros(len(y)), check["prior"], device="cpu")
            difference = {k: float(np.max(abs(v - decisions[k][:32]))) for k, v in reloaded.items()}
            if any(diff > (1e-6 if k.endswith("mode") else 2e-4) for k, diff in difference.items()):
                raise ValueError("checkpoint reload differs")
            if name == "weighted" and choice != report["choice"]:
                raise ValueError("pilot choice differs")
            title = kind + "_" + name
            result[title] = {"choice": choice, "metrics": m, "acceptance": acceptance(m, bm),
                             "reload_max_difference": difference, "selected_epoch": check["epoch"],
                             "weights_sha256": sha256_file(folder / "weights.pt"),
                             "predictions_sha256": sha256_file(folder / "predictions.npz")}
            rows.append(f"| {title} | {choice['strength']:.2f} | {choice['decision']} | {m['all']['rmse']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} |")
    budget = json.loads((base / "rarity_pilot_budget.json").read_text())
    if budget["state"] != "complete" or budget["returncode"] != 0 or budget["elapsed_seconds"] >= 1800:
        raise ValueError("pilot did not finish under its budget")
    output = {"verified": True, "scope": "single inner pilot only; selection and reported rows share the same inner labels",
              "test_access": False, "outer_labels_used": False, "baseline": bm, "models": result, "budget": budget,
              "goal_achieved": False}
    atomic_json(pilot / "verification.json", output)
    lines = ["# 稀缺标签加权：训练内小试验", "",
             "只使用outer0/inner1：训练排除0、1折，查询只取1折。表格是同一内层参与选模的探索性结果，没有独立外层成绩，不能据此宣称泛化提升。", "",
             f"匹配完整基线：整体RMSE {bm['all']['rmse']:.6f}，酸端{bm['acid']['rmse']:.6f}，碱端{bm['alkaline']['rmse']:.6f}。", "",
             "| 分支/训练目标 | 融合强度 | 解码 | 整体 RMSE | 酸端 RMSE | 碱端 RMSE |",
             "|---|---:|---|---:|---:|---:|", *rows, "",
             "加权先验只在训练子集计算；预测保存的原始训练标签先验与加权训练先验分开。权重是sigma0.5高斯标签密度的逆平方根，均值归一、截到8后再归一；因此最终最大权重可略超过8。网络、优化器和停止规则沿用自然目标，实际停止轮数可能不同。", "",
             f"本机任务实际{budget['elapsed_seconds']/60:.2f}分钟、硬上限30分钟。四个检查点均独立重载；使用虚拟查询标签也能复现预测。来源、输入、内层选择及融合指标已复核。", "",
             "两种加权分支均未通过既定双端门槛。不因这一小试验启动新的完整嵌套加权网格。当前完整自然目标实验继续按冻结协议运行；下一诊断优先区分随机投影/小隐层的信息损失与标签不足。"]
    (pilot / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verified": True, "models": {k: v["acceptance"] for k, v in result.items()}}, indent=2))


if __name__ == "__main__":
    main()
