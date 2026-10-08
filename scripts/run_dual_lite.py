"""Refit the full Dual-Lite recipe, freeze it, then evaluate matched PHOPT splits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phgeofuse.cache import atomic_json, atomic_text, sha256_file
from phgeofuse.config import get, load_config, path, save_resolved
from phgeofuse.delta_ref.metrics import metrics, paired_family_bootstrap
from phgeofuse.dual_fusion import retrieval_sequence_anchor
from phgeofuse.dual_lite import DualLiteFusion, load_sequence_features, write_predictions
from phgeofuse.io import read_fasta, read_manifest
from phgeofuse.lite.pipeline import fit as fit_lite, requested_records
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.robust_fusion import chemistry_features, pool_features
from phgeofuse.robust_train import fit as fit_robust, frequency_weights


def protected_files(config):
    files = [ROOT / name for name in (
        "phgeofuse/model.py", "phgeofuse/train.py", "phgeofuse/predict.py",
        "phgeofuse/dual_fusion.py", "phgeofuse/robust_fusion.py",
        "configs/phgeofuse_phopt.yaml")]
    files.extend(path(config, "paths." + name) for name in ("manifest", "retrieval", "folds"))
    for name in ("corrected_dual", "historical_dual"):
        directory = path(config, "paths." + name)
        bundle = directory / "bundle" if name == "corrected_dual" else directory
        files.extend(source for source in bundle.rglob("*") if source.is_file())
    return {str(source): sha256_file(source) for source in files}


def verify_protected(expected):
    actual = {source: sha256_file(source) for source in expected}
    if actual != expected:
        raise ValueError("an original model or source artifact changed during the run")


def check_official(records, split):
    suffix = {"train": "training", "validation": "validation", "test": "testing"}[split]
    official = read_fasta(ROOT / f"data/phopt_{suffix}.fasta", split)
    signature = lambda rows: sorted((row.protein_id, row.sequence, row.ph_opt) for row in rows)
    if signature(records) != signature(official):
        raise ValueError(f"manifest differs from official PHOPT {split}")


def read_control(source, records):
    with Path(source).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {row["key"]: row for row in rows}
    keys = [record_key(record) for record in records]
    if len(mapping) != len(rows) or set(mapping) != set(keys):
        raise ValueError(f"control prediction coverage differs: {source}")
    for record in records:
        row = mapping[record_key(record)]
        if not np.isclose(float(row["label"]), record.ph_opt, rtol=0, atol=1e-6):
            raise ValueError(f"control labels differ: {source}")
        if "sequence_sha256" in row and row["sequence_sha256"] != record.sequence_sha256:
            raise ValueError(f"control sequence differs: {source}")
    prediction = np.asarray([float(mapping[key]["prediction"]) for key in keys])
    if not np.isfinite(prediction).all():
        raise ValueError(f"nonfinite control predictions: {source}")
    return prediction


def evaluation_groups(config, records):
    with path(config, "paths.evaluation_clusters").open() as stream:
        pairs = list(csv.reader(stream, delimiter="\t"))
    mapping = {member: representative for representative, member in pairs}
    if len(mapping) != len(pairs):
        raise ValueError("duplicate evaluation cluster members")
    keys = [record_key(record) for record in records]
    if not set(keys) <= set(mapping):
        raise ValueError("evaluation clusters do not cover this split")
    return np.asarray([mapping[key] for key in keys])


def oof_inputs(config, training, features, output):
    keys = [record_key(record) for record in training]
    observed = np.asarray([record.ph_opt for record in training])
    rows = json.loads(path(config, "paths.folds").read_text())["rows"]
    if [row["key"] for row in rows] != [record.protein_id for record in training]:
        raise ValueError("training fold keys differ")
    folds, groups = np.asarray([row["fold"] for row in rows]), np.asarray([row["group"] for row in rows])
    retrieval, sequence = np.zeros((len(training), 15)), np.zeros(len(training))
    source = path(config, "paths.corrected_dual")
    retrieval_hash = sha256_file(path(config, "paths.retrieval"))
    audit = []
    for fold in sorted(set(folds)):
        held, fit = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        if set(groups[held]) & set(groups[fit]):
            raise ValueError("homology groups cross training folds")
        cache = source / f"fold{fold}.pt"
        payload = torch.load(cache, map_location="cpu")
        expected = {"query_keys": [keys[index] for index in held],
                    "reference_keys": [keys[index] for index in fit],
                    "retrieval_sha256": retrieval_hash}
        if payload.get("metadata") != expected or set(payload["rows"]) != set(expected["query_keys"]):
            raise ValueError(f"OOF retrieval provenance differs: {cache}")
        view = RetrievalStore(payload)
        retrieval[held] = np.asarray([view.features(keys[index]).numpy() for index in held])
        ridge = Ridge(alpha=float(get(config, "dual.sequence_alpha")), solver="cholesky")
        sequence[held] = ridge.fit(features[fit], observed[fit]).predict(features[held])
        audit.append({"fold": int(fold), "fit_count": len(fit), "held_count": len(held),
                      "retrieval_cache_sha256": sha256_file(cache), "groups_disjoint": True})
        print(f"DUAL_LITE_OOF fold={fold} complete", flush=True)
    with np.load(source / "expert_oof_inputs.npz", allow_pickle=False) as cached:
        if list(cached["keys"]) != keys or not np.array_equal(cached["fold"], folds):
            raise ValueError("historical expert OOF keys differ")
        if not np.array_equal(retrieval, cached["retrieval"]):
            raise ValueError("historical OOF retrieval values differ")
        difference = float(np.max(abs(sequence - cached["sequence"])))
        if difference > 1e-9:
            raise ValueError("refitted OOF sequence predictions differ from the fixed recipe")
    np.savez_compressed(output / "expert_oof_inputs.npz", keys=np.asarray(keys),
                        retrieval=retrieval, sequence=sequence, fold=folds)
    atomic_json(output / "oof_audit.json", {"folds": audit,
                "max_sequence_difference_vs_corrected_recipe": difference,
                "retrieval_reused_read_only": True, "sequence_heads_refitted": True})
    return retrieval, sequence


def fit(config, output):
    started = time.time()
    if output.exists():
        raise FileExistsError(f"use a new run directory: {output}")
    if get(config, "lite.design") != "P+R" or float(get(config, "dual.weight")) != .5:
        raise ValueError("this experiment fixes P+R and the original 0.5 Dual weight")
    records = requested_records(path(config, "paths.manifest"), {"train", "validation"})
    training = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    check_official(training, "train")
    check_official(validation, "validation")
    protected = protected_files(config)
    sources = [ROOT / "scripts/run_dual_lite.py", ROOT / "phgeofuse/dual_lite.py",
               ROOT / "phgeofuse/lite/model.py", ROOT / "phgeofuse/lite/features.py",
               ROOT / "phgeofuse/lite/pipeline.py", Path(config["_config_path"])]
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", {
        "primary": "full Dual with only the PHGeoFuse baseline replaced by Lite P+R",
        "fit_split": "train", "selection_split": "validation", "dual_weight": .5,
        "robust_weights": {"baseline": .5, "sequence": .25, "residual": .25},
        "test_used_for_selection": False, "seed": int(get(config, "dual.seed")),
        "sources": {str(source): sha256_file(source) for source in sources},
        "original_files": protected,
        "scope": "Frozen encoders; all final readouts refitted. Five grouped OOF inputs train the dual residual; not a nested full-model evaluation.",
        "test_status": "Previously viewed PHOPT test; follow-up comparison."})
    resolved = {**config, "paths": {name: str(path(config, "paths." + name)) for name in config["paths"]}}
    resolved["paths"]["dual_lite_run"] = str(output)
    save_resolved(resolved, output / "config.resolved.yaml")
    bundle = output / "bundle"
    print("DUAL_LITE_FIT phgeofuse_lite", flush=True)
    lite_result = fit_lite(config, output=bundle / "phgeofuse_lite")
    print("DUAL_LITE_FIT robust_readouts", flush=True)
    fit_robust(path(config, "paths.manifest"), path(config, "paths.retrieval"),
               path(config, "paths.esm2_features"), bundle / "robust_v1", seed=int(get(config, "dual.seed")))
    features = load_sequence_features(training, path(config, "paths.esm1v_features"),
                                     path(config, "paths.esm2_features"))
    x = np.column_stack([pool_features(*features[:2], "mean_std"),
                         pool_features(*features[2:], "mean_std")])
    observed = np.asarray([record.ph_opt for record in training])
    retrieval, sequence = oof_inputs(config, training, x, output)
    anchor = retrieval_sequence_anchor(retrieval, sequence)
    weights = frequency_weights(observed, .25)
    weights = np.clip(weights / weights.mean(), .25, 4.)
    residual = HistGradientBoostingRegressor(**get(config, "dual.residual"),
                                             random_state=int(get(config, "dual.seed")))
    residual.fit(np.column_stack([retrieval, sequence, chemistry_features([record.sequence for record in training])]),
                 observed - anchor, sample_weight=weights)
    sequence_model = Ridge(alpha=float(get(config, "dual.sequence_alpha")), solver="cholesky").fit(x, observed)
    joblib.dump(sequence_model, bundle / "sequence.joblib")
    joblib.dump(residual, bundle / "residual.joblib")
    provenance_sources = [path(config, "paths." + name) for name in
                          ("esm1v_features", "esm2_features", "folds")]
    provenance_sources.extend(path(config, "paths." + name).with_name("provenance.json")
                              for name in ("esm1v_features", "esm2_features"))
    metadata = {
        "format": "dual_lite", "schema_version": 1, "architecture": "Full Dual / PHGeoFuse-Lite P+R",
        "dual_weight": .5, "esm_dimensions": [features[0].shape[1], features[2].shape[1]],
        "seed": int(get(config, "dual.seed")), "training_count": len(training),
        "validation_count": len(validation), "labels_used": "PHOPT train only; validation selects Lite alpha",
        "test_used_for_selection": False, "sequence_alpha": sequence_model.alpha,
        "residual_recipe": residual.get_params(), "lite_alpha": lite_result["alpha"],
        "features": {str(source): sha256_file(source) for source in provenance_sources},
        "scope": "OOF inputs train the dual residual; no full-model outer-fold evaluation",
        "file_hashes": {str(source.relative_to(bundle)): sha256_file(source)
                        for source in bundle.rglob("*") if source.is_file()},
    }
    atomic_json(bundle / "model.json", metadata)
    atomic_json(output / "freeze.json", {"model_sha256": sha256_file(bundle / "model.json"),
                "protocol_sha256": sha256_file(output / "protocol.json"),
                "configuration_sha256": sha256_file(output / "config.resolved.yaml"),
                "seconds": time.time() - started, "test_scored": False})
    verify_protected(protected)
    print("DUAL_LITE_FIT frozen", flush=True)


def evaluate(config, output):
    if (output / "results.json").exists():
        raise FileExistsError("evaluation already recorded; refusing repeated test scoring")
    freeze = json.loads((output / "freeze.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    bundle = output / "bundle"
    if (freeze["model_sha256"] != sha256_file(bundle / "model.json")
            or freeze["protocol_sha256"] != sha256_file(output / "protocol.json")
            or freeze["configuration_sha256"] != sha256_file(output / "config.resolved.yaml")):
        raise ValueError("model or protocol changed after freeze")
    for source, expected in protocol["sources"].items():
        if sha256_file(source) != expected:
            raise ValueError(f"training source changed after freeze: {source}")
    verify_protected(protocol["original_files"])
    model = DualLiteFusion(bundle)
    store = RetrievalStore.load(path(config, "paths.retrieval"))
    result = {"model": "Dual-Lite", "primary": "fixed original full-Dual recipe",
              "model_sha256": freeze["model_sha256"], "seed": model.config["seed"],
              "training_seconds": freeze["seconds"], "splits": {}, "test_used_for_selection": False,
              "protocol": protocol["scope"], "test_status": protocol["test_status"]}
    for split in ("train", "validation", "test"):
        records = requested_records(path(config, "paths.manifest"), {split})
        check_official(records, split)
        groups = evaluation_groups(config, records)
        suffix = "_test_features" if split == "test" else "_features"
        prediction = model.predict_records(records, config, path(config, "paths.esm1v" + suffix),
                                           path(config, "paths.esm2" + suffix), store=store)
        observed = np.asarray([record.ph_opt for record in records])
        retrieval = np.asarray([store.features(record_key(record)).numpy() for record in records])
        low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
        controls = {}
        control_sources = {}
        if split != "train":
            control_sources["corrected_dual"] = path(config, "paths.corrected_dual") / f"{split}.csv"
            control_sources["historical_dual"] = (path(config, "paths.historical_test") if split == "test"
                                                   else path(config, "paths.historical_dual") / "seed42_validation.csv")
            controls = {name: read_control(source, records) for name, source in control_sources.items()}
        scores = {name: metrics(observed, values, groups, low) for name, values in {**prediction, **controls}.items()}
        summary = {"count": len(records), "metrics": scores,
                   "control_sha256": {name: sha256_file(source) for name, source in control_sources.items()}}
        if split != "train":
            baselines = {**controls, "phgeofuse_lite": prediction["phgeofuse_lite_prediction"]}
            summary["paired_cluster_bootstrap"] = paired_family_bootstrap(
                observed, prediction["prediction"][None], {name: values[None] for name, values in baselines.items()},
                groups, draws=int(get(config, "dual.bootstrap_draws")), seed=42)
        result["splits"][split] = summary
        write_predictions(output / f"{split}.csv", records, {**prediction, **controls}, include_labels=True)
        print("DUAL_LITE_EVALUATE", split, {name: value["all"]["rmse"] for name, value in scores.items()}, flush=True)
    verify_protected(protocol["original_files"])
    result["original_models_unchanged"] = True
    result["test_scored"] = True
    atomic_json(output / "results.json", result)
    write_report(output, result)
    return result


def write_report(output, result):
    lines = ["# Dual-Lite 训练与评测报告", "", "原 Dual、PHGeoFuse 神经模型和 Lite 单模产物均保留。",
             "冻结 SaProt/ESM1v/ESM2 表征；本次重新拟合读出头，不进行 GPU 编码器微调。", "",
             "结构：`0.5 × robust(Lite) + 0.5 × 双编码器专家`。",
             "`robust(Lite) = 0.5 × Lite + 0.25 × ESM2 Ridge + 0.25 × 检索/化学残差专家`。",
             "因此 Lite 在最终输出中的直接系数是 0.25，接入位置与旧神经 PHGeoFuse 相同。", "",
             "Lite 使用冻结 SaProt mean/std + 15 维检索特征 + Ridge；没有 EGNN、pH 解码器或学习式门控。",
             "本次保留原 Dual 的 Ridge/HGB 辅助专家及固定融合系数；它们不属于被移除的 PHGeoFuse 神经部件。", "",
             "## 训练协议", "", "7124 条 train 拟合，760 条 validation 只选择 Lite alpha；不合并 validation。",
             "双编码器残差头使用五折同源分组的折外检索和序列预测；每折检索缓存只读复用并校验参考/查询键。",
             "五个折内 Ridge 和最终 Ridge 均重新拟合。该折外输入训练不是完整模型的嵌套交叉验证成绩。",
             "训练完成后锁定整个包及配置，再评测 1971 条 test；测试集历史上已经被查看，属于后续比较。",
             "没有按 test 选择默认版本、融合系数或训练参数。", "",
             f"Lite alpha：{json.loads((output / 'bundle/model.json').read_text())['lite_alpha']}；",
             f"全部拟合耗时：{result['training_seconds']:.2f} 秒（已有冻结缓存，不含上游结构/编码）。", "",
             "## 整体指标", "", "| 模型 | Train RMSE | Validation RMSE | Test RMSE | Test MAE | Test R² |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    names = {"prediction": "新完整 Dual-Lite", "phgeofuse_lite_prediction": "Lite 单模 P+R",
             "robust_lite_prediction": "robust(Lite)", "dual_prediction": "双编码器专家",
             "corrected_dual": "旧完整 Dual（修复缓存，seed42）", "historical_dual": "历史完整 Dual（seed42）"}
    for name, label in names.items():
        values = [result["splits"][split]["metrics"].get(name) for split in ("train", "validation", "test")]
        scores = [f"{value['all']['rmse']:.6f}" if value else "未重新计算" for value in values]
        lines.append(f"| {label} | {' | '.join(scores)} | {values[2]['all']['mae']:.6f} | {values[2]['r2']:.6f} |")
    lines.extend(["", "## 测试集分层", "", "| 分组 | N | 新 Dual-Lite RMSE | 旧修复版 Dual RMSE | Lite 单模 RMSE |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    test = result["splits"]["test"]["metrics"]
    for name, label in (("acid", "pH ≤ 4"), ("core", "4 < pH < 10"),
                        ("alkaline", "pH ≥ 10"), ("low_homology", "低同源")):
        values = [test[key][name] for key in ("prediction", "corrected_dual", "phgeofuse_lite_prediction")]
        numbers = [f"{value['rmse']:.6f}" if value["rmse"] is not None else "无样本" for value in values]
        lines.append(f"| {label} | {values[0]['count']} | {' | '.join(numbers)} |")
    lines.extend(["", "低同源沿用 identity < 0.2 或任一比对覆盖率 < 0.8 的判断；不是全测试集同源隔离。",
                  "", "## 配对比较", "", "同一批样本按 30% 一致性、80% 双向覆盖率的 MMseqs 序列簇抽样 2000 次。",
                  "以下 Δ = 新 Dual-Lite RMSE − 对照 RMSE，负值表示新模型更好。", "",
                  "| 对照 | Test Δ RMSE | 簇级 bootstrap 95% CI |", "| --- | ---: | --- |"])
    comparisons = result["splits"]["test"]["paired_cluster_bootstrap"]["comparisons"]
    for name, label in (("corrected_dual", "旧修复版 Dual"), ("historical_dual", "历史 Dual"),
                        ("phgeofuse_lite", "Lite 单模")):
        baseline = name if name != "phgeofuse_lite" else "phgeofuse_lite_prediction"
        delta = test["prediction"]["all"]["rmse"] - test[baseline]["all"]["rmse"]
        interval = comparisons[name]["all"]["rmse"]["ci95"]
        lines.append(f"| {label} | {delta:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] |")
    lines.extend(["", "酸/碱端 RMSE、MAE 和偏差的多重比较校正区间也记录于 results.json。",
                  "一次 seed42 固定配方的结果不能代表跨种子或外部数据的稳定收益。",
                  "原模型哈希在训练前、训练后和评测后校验一致。", "",
                  "## 产物与复现", "", f"模型包：`{output / 'bundle'}`。",
                  "目录内有 protocol.json、freeze.json、oof_audit.json、config.resolved.yaml、",
                  "results.json 和 train/validation/test.csv。模型包包含 Lite JSON、robust 和 dual 读出头。", "",
                  "```bash", "python scripts/run_dual_lite.py --config configs/dual_lite_phopt.yaml", "```", "",
                  "再次实验须显式指定未使用的 `--output`，避免覆盖已锁定结果。",
                  "推理入口是 `python -m phgeofuse.dual_lite`，只使用准备好的 SaProt、ESM1v、ESM2 和检索输入，不读取查询标签。",
                  "SaProt 与检索仍需结构/3Di，完整 Dual-Lite 不是纯氨基酸序列模型。", ""])
    atomic_text(output / "REPORT_ZH.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dual_lite_phopt.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.expanduser().resolve() if args.output else path(config, "paths.dual_lite_run")
    torch.set_num_threads(8)
    if args.stage in ("fit", "all"):
        fit(config, output)
    if args.stage in ("evaluate", "all"):
        # Evaluation uses the configuration saved at fit time.
        evaluate(load_config(output / "config.resolved.yaml"), output)
    print(f"DUAL_LITE_COMPLETE stage={args.stage} output={output}", flush=True)


if __name__ == "__main__":
    main()
