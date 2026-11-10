import hashlib
import json
import platform
from importlib.metadata import version

import pandas as pd
from sklearn.metrics import f1_score

from .config import (
    CATEGORICAL, DATA_FILE_PATH, FEATURES, K_VALUES, OUTPUTS_DIR, PROJECT_PATH, RAW_DATA_DIR,
    SEED, TARGET,
)
from .data_io import load_data
from .evaluation import evaluate
from .inspection import inspect_data
from .model import MixedNaiveBayes, make_knn, make_lda
from .preprocessing import clean_records, split_records


def choose_k(X, y, manifest):
    train = manifest["split"].eq("train")
    validation = manifest["split"].eq("validation")
    scores = {}
    for k in K_VALUES:
        model = make_knn(k).fit(X.loc[train], y.loc[train])
        predicted = model.predict(X.loc[validation])
        scores[k] = float(f1_score(y.loc[validation], predicted))
    # F1 并列时选较小的 k，规则在查看测试集前固定。
    best = min(scores, key=lambda k: (-scores[k], k))
    return best, scores


def run_experiment():
    print("项目根目录：", PROJECT_PATH)
    print("原始数据目录：", RAW_DATA_DIR)
    print("输出目录：", OUTPUTS_DIR)
    raw = load_data(DATA_FILE_PATH)
    inspect_data(raw, categorical_columns=CATEGORICAL)
    print("数据对象类型", type(raw))
    print("数据形状：", raw.shape)
    print("字段名称：", raw.columns.tolist())
    print("前 5 行：")
    print(raw.head())

    records, audit = clean_records(raw)
    manifest = split_records(records)
    groups = manifest.groupby("group")
    if groups["split"].nunique().max() != 1 or manifest["split"].isna().any():
        raise RuntimeError("特征组划分不完整或存在跨集重叠。")
    audit["feature_groups"] = groups.ngroups
    audit["conflicting_groups"] = int(groups[TARGET].nunique().eq(2).sum())
    split_stats = {
        name: {"rows": len(part), "positive": int(part[TARGET].sum()),
               "groups": int(part["group"].nunique())}
        for name, part in manifest.groupby("split")
    }

    X, y = records[FEATURES], records[TARGET]
    best_k, validation_f1 = choose_k(X, y, manifest)
    models = {"bayes": MixedNaiveBayes(), "knn": make_knn(best_k), "lda": make_lda()}
    test = manifest["split"].eq("test")
    predictions = manifest.loc[test, ["source_row", TARGET]].copy()
    metrics = {}
    for name, model in models.items():
        model.fit(X.loc[~test], y.loc[~test])
        scores = model.predict_proba(X.loc[test])[:, 1]
        metrics[name], predictions[name] = evaluate(y.loc[test], scores)
        predictions[f"{name}_score"] = scores

    results = {
        "input_sha256": hashlib.sha256(DATA_FILE_PATH.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("pandas", "numpy", "scipy", "scikit-learn")
        }},
        "seed": SEED, "audit": audit, "splits": split_stats,
        "validation_f1": validation_f1, "best_k": best_k, "test_metrics": metrics,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUTPUTS_DIR / "cleaned_data.csv", index=False)
    manifest.to_csv(OUTPUTS_DIR / "splits.csv", index=False)
    predictions.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    (OUTPUTS_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"已删除吸烟史 No Info 记录 {audit['no_info_removed']} 条，去重后共 {len(records)} 条。")
    print(f"验证集选定 k = {best_k}")
    print(pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1", "fn", "fp"]].to_string())
    return results
