"""Recompute completed DeltaRef controls, preserving the user's LoRA pause."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import sha256_file, atomic_json
from phgeofuse.delta_ref.metrics import metrics, acceptance


def main():
    root = ROOT / "experiments/delta_ref_phopt_20260916"
    out = root / "analysis/controls_complete_20260917"
    out.mkdir(parents=True, exist_ok=True)
    source = np.load(root / "frozen/predictions.npz", allow_pickle=False)
    # The frozen stage stores labels as y and its nested selection as selected.
    keys = source["keys"].astype(str)
    names = source.files
    y = source["y"] if "y" in names else source["label"]
    groups = source["groups"]
    summaries = {"baseline": metrics(y, source["baseline"], groups)}
    files = [root / "frozen/predictions.npz"]
    for title, folder in [("DeltaRef", root / "frozen"), ("direct", root / "ablations/absolute"),
                          ("additive_difference", root / "ablations/additive")]:
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["keys"].astype(str), keys):
                raise ValueError("control keys mismatch")
            prediction_name = next(k for k in ("selected", "prediction", "nested_selected") if k in z.files)
            summaries[title] = metrics(y, z[prediction_name], groups)
        files.append(folder / "predictions.npz")
    fixed = root / "ablations/fixed_controls.npz"
    with np.load(fixed, allow_pickle=False) as z:
        if not np.array_equal(z["keys"].astype(str), keys):
            raise ValueError("fixed control keys mismatch")
        for name in ("natural_panel", "no_agreement", "no_baseline", "affine_expansion"):
            summaries[name] = metrics(y, z[name], groups)
    files.append(fixed)
    verdicts = {k: acceptance(v, summaries["baseline"]) for k, v in summaries.items() if k != "baseline"}
    payload = {"metrics": summaries, "acceptance": verdicts, "scope": "7124 training rows, grouped development only",
               "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
               "lora_started": False, "lora_status": "paused by user", "default_model_replaced": False}
    atomic_json(out / "verification.json", payload)
    lines = ["# ΔRef 已完成消融：本轮不继续 LoRA", "", "完整 7124 条训练样本的五外层/四内层同源分组开发结果，开发 seed42。用户已明确暂停 LoRA，本轮未启动或排程。", "",
             "| 方法 | 整体 RMSE | 整体 MAE | 极酸 RMSE | 极碱 RMSE | 验收 |", "|---|---:|---:|---:|---:|---|"]
    for name, m in summaries.items():
        status = "基线" if name == "baseline" else ("通过" if verdicts[name]["passed"] else "未通过")
        lines.append(f"| {name} | {m['all']['rmse']:.6f} | {m['all']['mae']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {status} |")
    lines += ["", "上述控制均沿用每个外层自己的内层选参。不能把使用全开发集选出的修正强度回填到外层成绩。逐样本预测已重新核验，来源哈希见 verification.json。此次结果不证明参考差值交互具有超越直接回归或可加差值的稳定贡献；不替换原模型。",
              "", "本次短时续跑完成了原先未完成的消融，总监督墙钟约21分钟。历史全部差值/直接/可加控制合计300次拟合，累计拟合耗时见顶层 run_costs.json。"]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
