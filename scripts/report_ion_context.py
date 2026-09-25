"""Independently recompute and summarize the completed local-context experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.metrics import metrics, acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    out = args.experiment
    result = json.loads((out / "results.json").read_text())
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    recomputed = {}
    for name, key in [("baseline", "baseline"), ("candidate", "prediction"), ("global_only_control", "global_only_control")]:
        recomputed[name] = metrics(arrays["y"], arrays[key], arrays["groups"])
        for group in ("all", "acid", "alkaline", "core"):
            for metric in ("rmse", "mae", "bias", "abs_bias"):
                if abs(recomputed[name][group][metric] - result[name][group][metric]) > 1e-12:
                    raise ValueError("stored summary differs from keyed predictions")
    verdict = acceptance(recomputed["candidate"], recomputed["baseline"])
    if verdict != result["acceptance"]:
        raise ValueError("acceptance mismatch")
    protocol = json.loads((out / "protocol.json").read_text())
    for filename, digest in protocol["source_hashes"].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError(f"source changed: {filename}")
    rows = []
    for r in result["outer"]:
        pair = {}
        for kind in ("selected", "global_only_control"):
            recipe = r[kind]["recipe"]["name"]
            m = json.loads((out / f"excluded_{r['outer']}" / f"{recipe}.json").read_text())
            tr, va = m["train_metrics"]["all"]["rmse"], m["query_metrics"]["all"]["rmse"]
            pair[kind] = {"train": tr, "heldout": va, "gap": va - tr, "recipe": recipe}
        rows.append({"outer": r["outer"], **pair})
    verification = {"recomputed": recomputed, "acceptance": verdict, "fit_gap_diagnostic": rows,
                    "source_hashes_verified": True,
                    "artifact_hashes": {name: sha256_file(out / name) for name in
                                        ("protocol.json", "results.json", "predictions.npz", "predictions.csv")}}
    atomic_json(out / "verification.json", verification)
    lines = ["# IonContext-pH 完整嵌套开发结果", "", "全部 7124 条 PHOPT 训练样本的五外层/四内层同源家族留出结果。确定性开发试验，未使用原始测试成绩选参，也不是五种子最终测试。", "",
             "| 模型 | 整体 RMSE | 整体 MAE | 极酸 RMSE | 极碱 RMSE | 中心 RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, title in [("baseline", "完整旧基线"), ("global_only_control", "全局表征对照"), ("candidate", "局部上下文候选")]:
        m = recomputed[name]
        lines.append(f"| {title} | {m['all']['rmse']:.6f} | {m['all']['mae']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {m['core']['rmse']:.6f} |")
    lines += ["", "内部验收：" + ("通过。仍需最终验证与同协议强基线。" if verdict["passed"] else "未通过，不替换默认模型。"),
              "失败项：" + ", ".join(verdict["failures"]), "", "## 外层独立选择", "",
              "| 外层 | 配方 | 修正强度 |", "|---|---|---:|"]
    for row in result["outer"]:
        lines.append(f"| {row['outer']} | {row['selected']['recipe']['name']} | {row['selected']['strength']} |")
    lines += ["", "## 过拟合诊断", "", "下表为所选独立 Ridge 支路的训练/留出 RMSE，不是混合后整套预测器的训练误差。差距含训练内乐观偏差、家族分布变化与训练量影响，不能单凭差距缩小断言过拟合解决。", "",
              "| 外层 | 全局对照训练 | 全局对照留出 | 新支路训练 | 新支路留出 |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        a, b = row["global_only_control"], row["selected"]
        lines.append(f"| {row['outer']} | {a['train']:.5f} | {a['heldout']:.5f} | {b['train']:.5f} | {b['heldout']:.5f} |")
    lines += ["", f"180 次拟合及选择用时约 {result['seconds'] / 60:.2f} 分钟（不含最终 bootstrap 写盘）。",
              "逐样本指标已独立复算，源码哈希一致；verification.json 保存核验结果。10000 次配对家族 bootstrap 的完整结果见 results.json。当前是多轮开发后的探索性证据，不具有独立外部确认含义。"]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
