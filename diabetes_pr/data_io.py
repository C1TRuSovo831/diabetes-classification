import pandas as pd
import numpy as np

from .config import BINARY, CONTINUOUS, FEATURES, TARGET


def load_data(file_path):
    data = pd.read_csv(file_path)
    if set(data.columns) != set([*FEATURES, TARGET]):
        raise ValueError("CSV 的字段与课程数据不一致，请检查文件和表头。")
    if data.isna().any().any():
        raise ValueError("输入含空值，请先确定处理策略。")
    if not np.isfinite(data[CONTINUOUS].to_numpy(dtype=float)).all():
        raise ValueError("连续字段必须是有限数值。")
    if not data[[*BINARY, TARGET]].isin([0, 1]).all().all() or data[TARGET].nunique() != 2:
        raise ValueError("疾病标记应为 0 或 1，且目标必须包含两类。")
    return data

