import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY, CATEGORICAL, CONTINUOUS, FEATURES, SEED, TARGET


def clean_records(raw):
    """先排除吸烟史未知记录，再按全部九列去重。保留原始索引。"""
    unknown = raw["smoking_history"].str.strip().str.casefold().eq("no info")
    eligible = raw.loc[~unknown]
    records = eligible.drop_duplicates().copy()
    audit = {
        "raw_rows": len(raw), "raw_positive": int(raw[TARGET].sum()),
        "no_info_removed": int(unknown.sum()),
        "no_info_positive": int(raw.loc[unknown, TARGET].sum()),
        "eligible_rows": len(eligible), "eligible_positive": int(eligible[TARGET].sum()),
        "duplicates_removed": len(eligible) - len(records),
        "clean_rows": len(records), "clean_positive": int(records[TARGET].sum()),
        "missing_cells": int(raw.isna().sum().sum()),
    }
    return records, audit


def split_records(records):
    """按八项输入分组；同一组的所有标签只进入同一个子集。"""
    group_ids = records.groupby(FEATURES, sort=False).ngroup()
    labels = records[TARGET].groupby(group_ids).agg(["min", "max"])
    # 三种组：仅阴性、仅阳性、两种标签均有。
    strata = labels["min"] + labels["max"]
    development, test = train_test_split(
        strata.index, test_size=0.2, stratify=strata, random_state=SEED
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=strata.loc[development], random_state=SEED
    )
    split_by_group = pd.Series(index=strata.index, dtype="str")
    for name, group_index in [("train", train), ("validation", validation), ("test", test)]:
        split_by_group.loc[group_index] = name
    manifest = pd.DataFrame({
        "source_row": records.index + 1,  # 从 1 开始，不含表头
        "group": group_ids,
        "split": group_ids.map(split_by_group),
        "diabetes": records[TARGET],
    }, index=records.index)
    return manifest


def make_encoder():
    """通过 Pipeline 在训练数据上拟合；独热编码保留类别之间的对称距离。"""
    return ColumnTransformer([
        ("continuous", StandardScaler(), CONTINUOUS),
        ("binary", "passthrough", BINARY),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     dtype=np.float64), CATEGORICAL),
    ])
