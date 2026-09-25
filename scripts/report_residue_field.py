"""Reconstruct nested decisions, audit all fits, and report residue-field results.

This reads completed artifacts only. It never trains or chooses a new recipe.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import stable_hash
from phgeofuse.delta_ref.metrics import metrics, acceptance
from localph.residue_training import choose

KINDS = ("direct", "global", "sparse")
NAMES = {"baseline": "Complete baseline", "direct": "Direct head + baseline",
         "global": "Global field + baseline", "sparse": "Sparse field + baseline",
         "nested_selected": "Nested selection"}


def read(path):
    return json.loads(path.read_text())


def same_metrics(stored, actual):
    for group in ("all", "core", "acid", "alkaline"):
        for key in ("rmse", "mae", "bias", "abs_bias", "count", "families"):
            if not np.isclose(stored[group][key], actual[group][key], atol=1e-12, rtol=0):
                raise ValueError(f"metric mismatch: {group}.{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ex = args.experiment
    protocol, result, budget = read(ex / "protocol.json"), read(ex / "results.json"), read(args.budget)
    if read(ex / "status.json")["state"] != "complete":
        raise ValueError("incomplete experiment")
    if budget["state"] != "complete" or budget["returncode"] != 0 or budget["elapsed_seconds"] >= 10 * 3600:
        raise ValueError("budget supervisor did not complete within the local limit")
    for field in ("source_hashes", "baseline_hashes"):
        for filename, digest in protocol[field].items():
            if sha256_file(ROOT / filename) != digest:
                raise ValueError(f"hash mismatch: {filename}")
    with np.load(ex / "predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    keys, y, folds, groups = (arrays[k] for k in ("keys", "y", "folds", "groups"))
    if len(set(keys)) != len(keys) or set(folds) != set(range(5)):
        raise ValueError("invalid final prediction coverage")
    key_to_index = {key: i for i, key in enumerate(keys)}
    recomputed = {k: metrics(y, arrays[k], groups) for k in NAMES}
    for name, m in recomputed.items():
        same_metrics(result[name], m)
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    fit_rows, selections = [], []

    def audit_fit(folder, expected_fit, expected_query):
        cert, report = read(folder / "fit.json"), read(folder / "complete.json")
        for name, digest in read(folder / "verified_complete.json")["hashes"].items():
            if sha256_file(folder / name) != digest:
                raise ValueError(f"fit hash mismatch: {folder}/{name}")
        fit, query = (np.array([key_to_index[k] for k in cert[name]]) for name in ("fit_keys", "query_keys"))
        if not np.array_equal(fit, expected_fit) or not np.array_equal(query, expected_query):
            raise ValueError("fit/query membership differs")
        if set(groups[fit]) & set(groups[query]):
            raise ValueError("family overlap")
        if cert["fit_labels_sha256"] != stable_hash(y[fit].tolist()) or cert["protocol_sha256"] != sha256_file(ex / "protocol.json"):
            raise ValueError("fit provenance differs")
        history = read(folder / "history.json")
        if cert["fixed_epochs"] is None:
            selected = min(history, key=lambda r: tuple(r["checkpoint_rank"]))["epoch"]
        else:
            selected = cert["fixed_epochs"]
        if report["selected_epoch"] != selected:
            raise ValueError("checkpoint selection differs")
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            predictions = {k: z[k].copy() for k in z.files}
        for name, prediction in predictions.items():
            m = metrics(y[query], prediction)
            for group in ("all", "core", "acid", "alkaline"):
                for metric in ("rmse", "mae"):
                    if not np.isclose(m[group][metric], report["query_metrics"][name][group][metric], atol=1e-12, rtol=0):
                        raise ValueError("fit metric mismatch")
        fit_rows.append({"path": str(folder), "selected_epoch": selected,
                         "epochs_run": report["epochs_run"], "parameters": report["parameters"],
                         "seconds": report["seconds"]})
        return predictions, report

    for outer in range(5):
        fit, query = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        outer_choices = {}
        with np.load(baseline_root / f"excluded_{outer}" / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["keys"], keys[query]) or not np.array_equal(z["prediction"], arrays["baseline"][query]):
                raise ValueError("outer baseline differs")
        for kind in KINDS:
            inner_base, inner_predictions, epochs = np.full(len(y), np.nan), {}, []
            for inner in sorted(set(range(5)) - {outer}):
                subfit = np.flatnonzero(~np.isin(folds, (outer, inner)))
                subquery = np.flatnonzero(folds == inner)
                p, report = audit_fit(ex / f"outer{outer}" / kind / f"inner{inner}", subfit, subquery)
                epochs.append(report["selected_epoch"])
                for name, values in p.items():
                    inner_predictions.setdefault(name, np.full(len(y), np.nan))[subquery] = values
                excluded = "_".join(map(str, sorted((outer, inner))))
                with np.load(baseline_root / f"excluded_{excluded}" / "predictions.npz", allow_pickle=False) as z:
                    lookup = dict(zip(z["keys"], z["prediction"]))
                    inner_base[subquery] = [lookup[k] for k in keys[subquery]]
            decision = choose(y[fit], inner_base[fit], {k: p[fit] for k, p in inner_predictions.items()})
            saved = result["outer"][outer]["models"][kind]
            if decision != saved["choice"] or saved["epochs"] != max(1, int(np.median(epochs))):
                raise ValueError("inner-only decision differs")
            p, report = audit_fit(ex / f"outer{outer}" / kind / "refit", fit, query)
            if report["selected_epoch"] != saved["epochs"]:
                raise ValueError("fixed refit duration differs")
            prediction = np.clip(arrays["baseline"][query] + decision["strength"] * (p[decision["decision"]] - arrays["baseline"][query]), 0, 14)
            if not np.array_equal(prediction, arrays[kind][query]):
                raise ValueError("outer blend differs")
            outer_choices[kind] = decision
            train_m = report["train_metrics"][decision["decision"]]
            query_m = report["query_metrics"][decision["decision"]]
            selections.append({"outer": outer, "kind": kind, "decision": decision["decision"],
                               "strength": decision["strength"], "epochs": report["selected_epoch"],
                               "train_rmse": train_m["all"]["rmse"], "heldout_rmse": query_m["all"]["rmse"],
                               "rmse_gap": query_m["all"]["rmse"] - train_m["all"]["rmse"]})
        selected = min(KINDS, key=lambda k: (*outer_choices[k]["rank"], KINDS.index(k)))
        if selected != result["outer"][outer]["selected_kind"] or not np.array_equal(arrays[selected][query], arrays["nested_selected"][query]):
            raise ValueError("nested kind selection differs")
    accepted = {k: acceptance(recomputed[k], recomputed["baseline"]) for k in (*KINDS, "nested_selected")}
    if accepted != result["acceptance"]:
        raise ValueError("acceptance differs")
    audit = {"all_75_fits_verified": len(fit_rows) == 75, "fit_rows": fit_rows, "selection": selections,
             "metrics": recomputed, "acceptance": accepted, "budget": budget,
             "source_hashes_verified": True, "baseline_hashes_verified": True,
             "nested_decisions_reconstructed": True, "family_isolation_verified": True,
             "predictions_sha256": sha256_file(ex / "predictions.npz"),
             "goal_achieved": False, "default_model_replaced": False, "test_access": False}
    atomic_json(out / "verification.json", audit)
    rows = ["# 残基 pH 响应场：完成结果", "",
            "以下是原PHOPT训练集7124条样本的五外层同源分组开发预测，seed42。不是原始1971条测试集成绩，也不是五个随机种子。ΔRef LoRA保持暂停。", "",
            "| 方法 | 整体 RMSE | 整体 MAE | 极酸 RMSE | 极碱 RMSE | 极酸偏差 | 极碱偏差 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in recomputed.items():
        rows.append(f"| {NAMES[name]} | {m['all']['rmse']:.6f} | {m['all']['mae']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {m['acid']['bias']:+.6f} | {m['alkaline']['bias']:+.6f} |")
    rows += ["", "极酸定义pH≤4，极碱pH≥10。完整指标含中心区、假极端率及各折样本量，见verification.json。", "",
             "## 内部验收与不确定性", ""]
    for name, a in accepted.items():
        rows.append(f"- {NAMES[name]}：{'通过开发门槛，仍需确认' if a['passed'] else '未通过'}；失败项：{', '.join(a['failures']) or '无'}。")
    rows += ["", "开发门槛为整体/中心RMSE及MAE最多增加0.01，假极端率最多增加0.005，两端RMSE/MAE至少改善10%、绝对偏差至少改善20%。该容差不等于严格性能不变。嵌套选择器的家族bootstrap区间保存在nested_v2/results.json；重复使用过的开发折不属于未见确认性证据。", "",
             "## 选模与过拟合诊断", "",
             "下表为独立新分支的训练/外层RMSE，融合强度由内层决定。间隙描述分支的拟合和跨家族泛化，不能据此推断整个系统相对旧基线已减少过拟合。", "",
             "| 外层 | 分支 | 决策 | 融合强度 | 重训轮数 | 分支训练 RMSE | 分支外层 RMSE | 差值 |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in selections:
        rows.append(f"| {r['outer']} | {r['kind']} | {r['decision']} | {r['strength']:.2f} | {r['epochs']} | {r['train_rmse']:.6f} | {r['heldout_rmse']:.6f} | {r['rmse_gap']:+.6f} |")
    rows += ["", "每个内层的history.json保留逐轮训练损失和验证目标。响应场的NLL混合损失不能直接与普通回归的MSE数值比较；评估使用统一pH误差。小参数量本身不证明缓解过拟合。", "",
             "## 可复算与计算约束", "",
             f"完成75次拟合；整次任务{budget['elapsed_seconds']/60:.2f}分钟，硬上限{budget['limit_seconds']/3600:.1f}小时。没有启动LoRA。",
             "全部来源和基线哈希、75个拟合文件哈希、训练/查询家族隔离、检查点轮次、内层选择、外层融合与逐样本指标均重新核验。重载数值核验另存reload_verification*.json。中断的早期nested/不计入本结果。", "",
             "模型为统计pH响应场，不是物理酶活曲线、pKa或校准置信区间。设计和已检查文献的差异见docs/residue_field_phopt_20260917.md；目前没有领域首次或两端领先的证据。最终融合仍依赖旧结构/检索基线，应披露其信息来源。", "",
             "原目标尚未完成，默认模型没有替换。只有开发达标后才考虑冻结方案的原PHOPT验证、相同五种子对照及独立确认。"]
    (out / "REPORT_ZH.md").write_text("\n".join(rows) + "\n")
    plot(ex, out, y, arrays, selections)
    print(json.dumps({"fits_verified": len(fit_rows), "acceptance": accepted, "report": str(out / "REPORT_ZH.md")}, indent=2))


def plot(ex, out, y, arrays, selections):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    labels = ["pH <= 4", "4 < pH <= 6", "6 < pH <= 8", "8 < pH < 10", "pH >= 10"]
    masks = [y <= 4, (y > 4) & (y <= 6), (y > 6) & (y <= 8), (y > 8) & (y < 10), y >= 10]
    for name in ("baseline", "nested_selected", "sparse"):
        bias = [np.mean(arrays[name][m] - y[m]) for m in masks]
        axes[0].plot(range(5), bias, "o-", label=NAMES[name])
    axes[0].axhline(0, c="gray", lw=.8)
    axes[0].set(xticks=range(5), xticklabels=labels, ylabel="Prediction - label (pH)", title="Held-out shrinkage by true pH")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=7)
    for kind in KINDS:
        rows = [r for r in selections if r["kind"] == kind]
        axes[1].scatter([r["train_rmse"] for r in rows], [r["heldout_rmse"] for r in rows], label=kind)
    limits = axes[1].get_xlim()
    axes[1].plot(limits, limits, c="gray", lw=.8)
    axes[1].set(xlabel="Head training RMSE", ylabel="Head family-held-out RMSE", title="Unblended selected heads")
    axes[1].legend(fontsize=8)
    for kind in KINDS:
        histories = [read(p) for p in (ex / "outer0" / kind).glob("inner*/history.json")]
        length = max(map(len, histories))
        matrix = np.full((len(histories), length), np.nan)
        for i, history in enumerate(histories):
            matrix[i, :len(history)] = [r["standalone_validation_objective"] for r in history]
        axes[2].plot(np.arange(1, length + 1), np.nanmean(matrix, axis=0), label=kind)
    axes[2].set(xlabel="Epoch", ylabel="Validation MSE + 0.05 tail MSEs", title="Outer 0: active inner fits only")
    axes[2].legend(fontsize=8)
    fig.savefig(out / "diagnostics.png", dpi=180)
    fig.savefig(out / "diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
