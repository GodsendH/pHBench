"""Verify the full-width pilot and report its matched compression controls."""
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
from phgeofuse.delta_ref.data import DevelopmentData, stable_hash
from phgeofuse.delta_ref.metrics import metrics, acceptance


def main():
    base = ROOT / "experiments/residue_field_phopt_20260917"
    pilot = base / "capacity_pilot"
    protocol = json.loads((pilot / "protocol.json").read_text())
    for section in ("source_hashes", "baseline_hashes"):
        for name, digest in protocol[section].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError("capacity pilot provenance changed")
    for name, digest in protocol["feature_hashes"].items():
        if sha256_file(base / "full_features" / name) != digest:
            raise ValueError("full feature hash differs")
    if json.loads((pilot / "status.json").read_text())["state"] != "complete":
        raise ValueError("capacity pilot incomplete")
    budget = json.loads((base / "capacity_pilot_budget.json").read_text())
    if budget["state"] != "complete" or budget["returncode"] != 0 or budget["elapsed_seconds"] >= 1800:
        raise ValueError("capacity pilot did not finish within budget")
    d = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    t = d.train
    keys, y, folds, groups = d.keys[t], d.labels[t], d.folds[t], d.groups[t]
    fit, query = np.flatnonzero(~np.isin(folds, [0, 1])), np.flatnonzero(folds == 1)
    if (protocol["fit_keys"] != keys[fit].tolist() or protocol["query_keys"] != keys[query].tolist()
            or protocol["fit_labels_sha256"] != stable_hash(y[fit].tolist()) or set(groups[fit]) & set(groups[query])):
        raise ValueError("capacity pilot isolation differs")
    with np.load(base / "full_features/index.npz", allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("full feature keys differ")
        packed = {k: z[k].copy() for k in ("ionizable", "offsets")}
    packed["tokens"] = np.load(base / "full_features/tokens.npy", allow_pickle=False, mmap_mode="r")
    bdir = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42/excluded_0_1"
    with np.load(bdir / "predictions.npz", allow_pickle=False) as z:
        lookup = dict(zip(z["keys"], z["prediction"]))
        baseline = np.array([lookup[k] for k in keys[query]])
    bm = metrics(y[query], baseline, groups[query])
    results, rows, gaps = {}, [], []
    torch.set_num_threads(2)
    targets = [("128 natural", base / "nested_v2/outer0/sparse/inner1", 128),
               ("128 weighted", base / "rarity_pilot/sparse", 128),
               ("1280 natural", pilot / "natural", 1280), ("1280 weighted", pilot / "weighted", 1280)]
    small_verified = json.loads((base / "rarity_pilot/verification.json").read_text())
    for name, folder, width in targets:
        report = json.loads((folder / "complete.json").read_text())
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            decisions = {k: z[k].copy() for k in z.files if k != "keys"}
        choice = choose(y[query], baseline, decisions)
        fused = np.clip(baseline + choice["strength"] * (decisions[choice["decision"]] - baseline), 0, 14)
        m = metrics(y[query], fused, groups[query])
        check = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
        if width == 1280:
            model = ResidueField(kind="sparse", input_dim=1280)
            model.load_state_dict(check["state_dict"])
            reloaded = predict(model, packed, query[:32], np.zeros(len(y)), check["prior"], device="cpu")
            difference = {k: float(np.max(abs(v - decisions[k][:32]))) for k, v in reloaded.items()}
            if any(diff > (1e-6 if k.endswith("mode") else 2e-4) for k, diff in difference.items()):
                raise ValueError("full checkpoint reload differs")
            if choice != report["choice"]:
                raise ValueError("capacity choice differs")
            history = json.loads((folder / "history.json").read_text())
            if min(history, key=lambda r: tuple(r["checkpoint_rank"]))["epoch"] != check["epoch"]:
                raise ValueError("capacity checkpoint selection differs")
        else:
            old_name = "sparse_" + name.split()[1]
            previous = small_verified["models"][old_name]
            if sha256_file(folder / "weights.pt") != previous["weights_sha256"] or sha256_file(folder / "predictions.npz") != previous["predictions_sha256"]:
                raise ValueError("previously reloaded control changed")
            difference = previous["reload_max_difference"]
        train_m = report["train_metrics"]["prior1_mean"]
        query_m = report["query_metrics"]["prior1_mean"]
        natural_metrics = metrics(y[query], decisions["prior1_mean"], groups[query])
        for group in ("all", "acid", "alkaline", "core"):
            for key in ("rmse", "mae"):
                if not np.isclose(natural_metrics[group][key], query_m[group][key], atol=1e-12, rtol=0):
                    raise ValueError("capacity standalone metric differs")
        result = {"choice": choice, "metrics": m, "acceptance": acceptance(m, bm),
                  "natural_prior_head_train": train_m, "natural_prior_head_query": query_m,
                  "reload_max_difference": difference, "selected_epoch": check["epoch"],
                  "parameters": report.get("parameters", 7309),
                  "weights_sha256": sha256_file(folder / "weights.pt"),
                  "predictions_sha256": sha256_file(folder / "predictions.npz")}
        results[name] = result
        rows.append(f"| {name} | {result['parameters']} | {choice['strength']:.2f} | {m['all']['rmse']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} |")
        gaps.append(f"| {name} | {check['epoch']} | {train_m['all']['rmse']:.6f} | {query_m['all']['rmse']:.6f} | {query_m['all']['rmse'] - train_m['all']['rmse']:+.6f} | {train_m['acid']['rmse']:.6f} / {query_m['acid']['rmse']:.6f} | {train_m['alkaline']['rmse']:.6f} / {query_m['alkaline']['rmse']:.6f} |")
    atomic_json(pilot / "verification.json", {"verified": True, "scope": "one inner-fold diagnostic; no independent outer estimate",
                "models": results, "baseline": bm, "budget": budget, "goal_achieved": False,
                "test_access": False, "query_labels_required_for_reload": False})
    lines = ["# 残基表征容量诊断", "",
             "同一outer0/inner1、seed42；训练4274条、内层查询1425条。各方法在这些内层查询上选择检查点与融合，因此这些分数含选模偏差。没有使用outer0标签或原PHOPT验证/测试成绩选模。", "",
             "1280维分支使用全部残基的完整ESM1v表征，存储为float16；128维分支使用float32表征经过固定正交投影后存为float16。两者的隐层宽32、优化器、dropout、早停规则相同。输入维度增大也增加输入映射参数，因此不是参数量严格匹配的信息损失归因实验。", "",
             "## 融合后的内层误差", "",
             f"完整基线：整体{bm['all']['rmse']:.6f}，酸端{bm['acid']['rmse']:.6f}，碱端{bm['alkaline']['rmse']:.6f}。", "",
             "| 输入/目标 | 参数量 | 融合强度 | 整体 RMSE | 酸端 RMSE | 碱端 RMSE |",
             "|---|---:|---:|---:|---:|---:|", *rows, "",
             "## 独立分支的自然先验均值诊断", "",
             "统一使用prior1_mean，避免把不同后处理的误差差值当作参数容量效应。差值为留出RMSE减训练RMSE；不同训练规模及家族分布也会影响这一差值，不能直接视为过拟合的无偏估计。", "",
             "| 输入/目标 | 检查点轮数 | 训练 RMSE | 内层留出 RMSE | 差值 | 酸端训练 / 留出 | 碱端训练 / 留出 |",
             "|---|---:|---:|---:|---:|---:|---:|", *gaps, "",
             "## 验证与结论边界", "",
             f"全维双目标试验共{budget['elapsed_seconds']/60:.2f}分钟，硬上限30分钟；全维缓存单独受15分钟上限约束。两个全维检查点均以虚拟标签在CPU重载，较小分支复核先前重载文件哈希。源码、特征、家族隔离、检查点选择、融合与误差均重新核对。", ""]
    for name, r in results.items():
        a = r["acceptance"]
        lines.append(f"- {name}：{'达到内部数值门槛，仍需独立外层确认' if a['passed'] else '未通过双端内部门槛'}；失败项：{', '.join(a['failures']) or '无'}。")
    lines += ["", "该小试验不能证明原PHOPT性能维持、跨家族泛化改善或领域领先；完整自然目标5×4嵌套实验仍由独立冻结协议管理，不能将两者拼成新的无偏成绩。"]
    (pilot / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verified": True, "acceptance": {k: r["acceptance"] for k, r in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
