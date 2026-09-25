"""Verify the bounded 2026-09-17 iteration and retain a consolidated report."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.metrics import metrics, acceptance


def verify_summary(expected, actual):
    for group in ("all", "acid", "alkaline", "core"):
        for metric in ("rmse", "mae", "bias", "abs_bias"):
            if abs(expected[group][metric] - actual[group][metric]) > 1e-12:
                raise ValueError(f"recomputation differs: {group}.{metric}")


def main():
    ext = ROOT / "experiments/ion_context_external_phopt_20260917/nested"
    residual = ROOT / "experiments/ion_context_residual_phopt_20260917/results"
    site = ROOT / "experiments/ion_context_phopt_20260917/nested"
    density = ROOT / "experiments/ph_density_phopt_20260917/nested"
    out = ROOT / "experiments/extreme_ph_iteration_20260917"
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((ext / "results.json").read_text())
    protocol = json.loads((ext / "protocol.json").read_text())
    with np.load(ext / "predictions.npz", allow_pickle=False) as z:
        y, groups, keys = z["y"].copy(), z["groups"].copy(), z["keys"].copy()
        recomputed = {name: metrics(y, z[name], groups) for name in ("baseline", "candidate", "PHOPT_only", "external_global")}
    for name, value in recomputed.items():
        verify_summary(result[name], value)
    for field in ("source_hashes", "input_hashes", "baseline_source_hashes"):
        for filename, digest in protocol[field].items():
            if sha256_file(ROOT / filename) != digest:
                raise ValueError(f"hash differs: {filename}")
    a = acceptance(recomputed["candidate"], recomputed["baseline"])
    if a != result["acceptance"]:
        raise ValueError("acceptance differs")
    gaps = []
    for outer in result["outer"]:
        row = {"outer": outer["outer"]}
        for kind in ("candidate", "PHOPT_only", "external_global"):
            choice = outer[kind]
            p = ext / f"excluded_{outer['outer']}" / (choice["recipe"]["name"] + ".json")
            fit = json.loads(p.read_text())
            row[kind] = {"recipe": choice["recipe"], "strength": choice["strength"],
                         "train_rmse": fit["train_metrics"]["all"]["rmse"],
                         "heldout_rmse": fit["query_metrics"]["all"]["rmse"],
                         "external_offset": fit["external_offset"]}
        gaps.append(row)
    er = {"verified": True, "metrics": recomputed, "acceptance": a, "gaps": gaps,
          "test_access": False, "source_and_input_hashes_verified": True,
          "prediction_sha256": sha256_file(ext / "predictions.npz")}
    atomic_json(ext / "verification.json", er)
    rr = json.loads((residual / "results.json").read_text())
    with np.load(residual / "predictions.npz", allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys) or not np.array_equal(z["y"], y):
            raise ValueError("residual experiment alignment differs")
        for name, row in rr["fixed_recipes"].items():
            verify_summary(row["metrics"], metrics(y, z[name], groups))
    atomic_json(residual / "verification.json", {"verified": True, "recipes": len(rr["fixed_recipes"]),
                "predictions_sha256": sha256_file(residual / "predictions.npz"),
                "warning": "fixed exploratory recipes; no valid independent estimate for best row selected after viewing outer results"})
    sr = json.loads((site / "results.json").read_text())
    dr = json.loads((density / "results.json").read_text())
    with np.load(density / "predictions.npz", allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys) or not np.array_equal(z["y"], y):
            raise ValueError("density experiment alignment differs")
        for name in ("baseline", "candidate", "density_mean", "no_prior_correction"):
            verify_summary(dr[name], metrics(y, z[name], groups))
    dp = json.loads((density / "protocol.json").read_text())
    for filename, digest in dp["source_hashes"].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError("density source differs")
    atomic_json(density / "verification.json", {"verified": True, "source_hashes_verified": True,
                "predictions_sha256": sha256_file(density / "predictions.npz"),
                "acceptance": acceptance(dr["candidate"], dr["baseline"])})
    budget_paths = [
        ROOT / "experiments/delta_ref_phopt_20260916/controls_budget_20260917.json",
        ROOT / "experiments/ion_context_phopt_20260917/cache_budget.json",
        ROOT / "experiments/ion_context_phopt_20260917/nested_budget.json",
        ROOT / "experiments/ion_context_phopt_20260917/external_data_audit/budget2.json",
        ROOT / "experiments/ion_context_external_phopt_20260917/cache_budget.json",
        ROOT / "experiments/ion_context_external_phopt_20260917/nested_budget.json",
        ROOT / "experiments/ion_context_residual_phopt_20260917/budget.json",
        ROOT / "experiments/ph_density_phopt_20260917/budget.json"]
    budgets = [json.loads(p.read_text()) for p in budget_paths]
    if any(b["state"] != "complete" or b["returncode"] != 0 or b["elapsed_seconds"] >= 10 * 3600 for b in budgets):
        raise ValueError("a bounded task is not successfully complete or exceeded 10 hours")
    rows = [("完整旧基线", recomputed["baseline"]), ("局部表征，仅PHOPT", sr["candidate"]),
            ("同精度PHOPT对照", recomputed["PHOPT_only"]),
            ("额外监督，全局对照", recomputed["external_global"]),
            ("额外监督，来源校正候选", recomputed["candidate"]),
            ("连续pH条件密度候选", dr["candidate"])]
    lines = ["# 极端 pH 模型迭代记录：2026-09-17", "",
             "本轮使用原 PHOPT 训练集内部的同源分组评估，没有新增原始测试集选参或测试结果。所有下表结果均为同一7124条训练样本的外层留出预测；不能与原始1971条测试RMSE直接相减。", "",
             "## 完整嵌套实验", "", "| 方法 | 整体 RMSE | MAE | 极酸 RMSE | 极碱 RMSE | 中心 RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for title, m in rows:
        lines.append(f"| {title} | {m['all']['rmse']:.6f} | {m['all']['mae']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {m['core']['rmse']:.6f} |")
    lines += ["", "来源校正候选内部验收：" + ("通过，仍需后续验证。" if a["passed"] else "未通过，保留旧模型。"),
              "失败项：" + ", ".join(a["failures"]), "",
              "每个外层均从其四个内层选择配方与修正强度。全局/局部、仅PHOPT/额外监督对照使用同一套选择准则。Ridge是确定性的，这些不是五次独立随机训练。完整家族bootstrap见外部版本nested/results.json；这是探索性开发，不能解释为未见外部数据的确认。", "",
              "## 外部数据与来源差异", "",
              "从EnzyBase12k公开元数据11615条出发，排除PHOPT accession/完全相同序列、截短域、超出初始1022残基合同及无效记录，得到2744条。进一步对全部PHOPT序列作MMseqs搜索，排除检出的>=30% identity且双向覆盖>=80%命中，得到1119条，其中极酸27、极碱16。另一个独立核验脚本检查了13894条合格命中和每行原始出处，未读取PHOPT验证/测试标签。启发式未命中不等于无生物学同源。", "",
              "PHOPT和外部ESM1v均为float32、无TF32的匹配表征；ESM2均使用float16模型与float32池化。外部监督版本增加正则化的来源偏移系数，预测PHOPT时来源指示固定为零。该方案没有额外模型路由或支持集适配，但来源偏移假设仅覆盖常数差异，不保证解决底物/条件差异。", "",
              "## 固定残差配方试验", "",
              "另完成8个固定正则/权重配方×5外层，报告2个固定修正强度，共16行。训练残差来自同时排除外层与内层的完整基线；未使用外层标签训练。这是探索性固定配方评估，不能把事后最佳行当成新的嵌套选模成绩。", "",
              "| 固定配方 | 整体 RMSE | 极酸 RMSE | 极碱 RMSE | 内部通过 |", "|---|---:|---:|---:|---|"]
    for name, r in rr["fixed_recipes"].items():
        m = r["metrics"]
        lines.append(f"| {name} | {m['all']['rmse']:.6f} | {m['acid']['rmse']:.6f} | {m['alkaline']['rmse']:.6f} | {r['acceptance']['passed']} |")
    lines += ["", "## 连续pH条件密度试验", "",
              "另以24个余弦矩表达连续pH标签，使用共享的正则化回归，沿pH轴作高斯平滑。内层比较正则0.2/2、平滑宽度0.5/1、训练先验校正指数0/0.5、均值/峰值两种决策，共16配方、30次密度拟合。pH密度的负值截断，记录截断前负质量；该分布不是校准置信区间或真实活性曲线。所有监督、边缘先验和融合强度均只使用当前训练子集。", "",
              "五个外层均选择修正强度0，退回完整基线；均值对照和不校正先验对照同样如此。该结果排除本轮固定表征/线性条件密度方案，不能证明所有分布预测方法无效。实现/预测/来源核验位于experiments/ph_density_phopt_20260917/nested/。", "",
              "## ΔRef状态", "",
              "300次历史/本轮冻结差值、直接及可加控制拟合已完成，逐样本消融全部复算。ΔRef及消融均未通过内部双端验收。用户明确暂停LoRA，本轮未启动或安排LoRA集群任务。完成消融报告位于experiments/delta_ref_phopt_20260916/analysis/controls_complete_20260917/REPORT_ZH.md。", "",
              "## 本机计算与可复算性", "", "| 任务 | 实际分钟 | 硬上限分钟 |", "|---|---:|---:|"]
    for b in budgets:
        lines.append(f"| {Path(b['command'][3]).name if len(b['command']) > 3 else b['command'][-1]} | {b['elapsed_seconds']/60:.2f} | {b['limit_seconds']/60:.0f} |")
    lines += ["", "所有已启动任务正常结束且低于10小时。源码、数据、基线排除缓存和逐样本预测均有哈希；单元测试检查池化/缺失类型、标准化隔离、Ridge数值一致性、零外部权重、来源偏移及进程组超时。没有使用本轮新候选覆盖生产模型。", "",
              "## 当前结论与下一步", "",
              "这轮没有建立双端领先或过拟合已解决的证据。固定冻结表征上的均方损失回归、局部统计、少量额外监督、线性残差与线性条件密度均未达标。下一阶段优先研究能跨家族迁移的残基层功能信息和不同监督目标，不继续单纯扩大当前线性/均值池化配方网格；保留同容量连续回归、来源/权重消融。先在训练内开发，达到门槛后再启动五种子完整系统验证。任何新结构仍须和领域文献对照，不能预先承诺一定成功或宣称首创。"]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "status.json", {"phase": "bounded_iteration_complete", "goal_achieved": False,
                "lora_paused_by_user": True, "default_model_replaced": False, "all_jobs_completed": True,
                "external_acceptance": a, "new_local_features_acceptance": sr["acceptance"],
                "density_acceptance": dr["acceptance"],
                "fixed_residual_passes": [k for k, r in rr["fixed_recipes"].items() if r["acceptance"]["passed"]],
                "budget_records": budgets})
    print("\n".join(lines[:17]))


if __name__ == "__main__":
    main()
