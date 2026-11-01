# 糖尿病分类预测：贝氏分类、k-NN 与 LDA 比较

模式识别课程实作，使用 Python 对课程提供的糖尿病预测数据进行处理，计划比较贝氏分类、k-最近邻法（k-NN）和线性判别分析（LDA）三种方法。

> **当前阶段：项目架构与启动验证。** 目前入口程序仅输出项目路径，尚未实现数据读取、预处理、模型训练或评估。本说明中的后续流程属于开发计划，不代表已经得到实验结果。

## 1. 任务目标

- 根据个体健康相关特征，预测 `diabetes` 对应的类别。
- 将数据读取、预处理、模型构建、评估和流程协调分开组织，避免把全部逻辑写在一个脚本中。
- 在一致的数据划分与评估条件下，比较三种分类方法。
- 记录数据处理方式、实验设置和结果，支持后续复现与分析。

`diabetes` 是目标标签，不能同时作为输入特征。贝氏分类的具体实现与假设将在结合课程内容和数据特点后明确；当前尚未选定或实现模型。

## 2. 当前进度

- [x] 建立 Conda 环境 `pr_diabetes`，使用 Python 3.11。
- [x] 建立项目包、入口和路径配置。
- [x] 将原始 CSV 放入数据目录。
- [x] 从项目根目录运行入口，正确输出三个路径。
- [ ] 实现 CSV 读取函数，并检查行列数、列名和样本内容。
- [ ] 检查字段类型、缺失值、重复记录和标签分布。
- [ ] 明确输入特征与标签，制定数据划分方案。
- [ ] 实现与验证预处理流程。
- [ ] 实现三种分类方法及统一实验流程。
- [ ] 整理评估指标、图表和比较分析。
- [ ] 完成课程要求的代码归档和实作报告。

## 3. 项目结构

```text
diabetes_classification/
├── main.py                    # 启动入口
├── README.md                  # 项目说明
├── diabetes_pr/               # 项目自己的 Python 包
│   ├── __init__.py            # 包标识文件
│   ├── config.py              # 项目路径与后续实验设置
│   ├── data_io.py             # 数据读取模块，待实现
│   ├── preprocessing.py       # 预处理模块，待实现
│   ├── model.py               # 模型构建模块，待实现
│   ├── evaluation.py          # 评估模块，待实现
│   └── experiment.py          # 实验流程协调，目前仅输出路径
├── data/
│   └── raw/
│       └── diabetes_prediction_dataset.csv
└── outputs/                   # 预留的实验输出目录，目前无结果
```

### 模块职责

| 文件 | 职责与当前状态 |
| --- | --- |
| [main.py](<main.py>) | 导入并调用 `run_experiment()`；保持入口简洁。 |
| [config.py](<diabetes_pr/config.py>) | 定义 `PROJECT_PATH`、`RAW_DATA_DIR`、`OUTPUTS_DIR`，根据模块位置定位项目根目录。 |
| [data_io.py](<diabetes_pr/data_io.py>) | 计划接收文件路径并返回数据表格；目前只有库导入，尚无读取函数。 |
| [preprocessing.py](<diabetes_pr/preprocessing.py>) | 计划定义缺失值处理、类别编码和数值缩放等预处理；当前为空文件。 |
| [model.py](<diabetes_pr/model.py>) | 计划集中创建分类模型；当前为空文件。 |
| [evaluation.py](<diabetes_pr/evaluation.py>) | 计划根据真实标签与预测结果计算指标、整理结果；当前为空文件。 |
| [experiment.py](<diabetes_pr/experiment.py>) | 负责协调各模块；目前只打印路径，不执行完整实验。 |

约定：包内模块可使用相对导入，例如 `from .config import RAW_DATA_DIR`；项目入口使用完整包名导入。不通过修改 `sys.path` 来绕过导入问题。

## 4. 数据说明

原始数据为课程提供的 [diabetes_prediction_dataset.csv](<data/raw/diabetes_prediction_dataset.csv>)。作业说明标注其规模为 **100,000 条样本、9 个字段**.

| 字段 | 含义 | 角色 |
| --- | --- | --- |
| `gender` | 性别类别 | 候选输入特征 |
| `age` | 年龄 | 候选输入特征 |
| `hypertension` | 高血压相关标记 | 候选输入特征 |
| `heart_disease` | 心脏病相关标记 | 候选输入特征 |
| `smoking_history` | 吸烟史类别 | 候选输入特征 |
| `bmi` | 身体质量指数 | 候选输入特征 |
| `HbA1c_level` | 糖化血红蛋白水平 | 候选输入特征 |
| `blood_glucose_level` | 血糖水平 | 候选输入特征 |
| `diabetes` | 糖尿病类别标记 | 目标标签 |

### 原始数据管理约定

- 原始文件仅用于读取，不通过清洗代码覆盖保存。
- 解压得到、内容未修改的 CSV 仍属于原始数据。
- 处理后的数据可以先保留在内存中；确需保存时，另设输出位置，不覆盖原件。
- 不因为发现缺失值或重复记录就直接删除，先明确其含义和处理依据。

## 5. 开发环境

### 已有环境

当前使用 Conda 环境 `pr_diabetes`，Python 版本为 3.11。

在能够使用 Conda 的 PowerShell 或 Anaconda Prompt 中激活环境：

```powershell
conda activate pr_diabetes
```

检查实际使用的 Python 和 pip：

```powershell
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

解释器路径通常应包含 `envs\pr_diabetes\python.exe`。如果使用 VS Code，还需通过 `Python: Select Interpreter` 选择同一环境；编辑器和终端的环境应分别核对。

### 在新机器上准备基础环境

已经创建环境的机器无需重复执行创建命令。新机器可以从以下基础环境开始：

```powershell
conda create --name pr_diabetes python=3.11 pip
conda activate pr_diabetes
python -m pip install pandas numpy
```

- `pathlib` 属于 Python 标准库，无需额外安装。
- `pandas` 用于 CSV 读取和表格检查。
- `numpy` 用于后续数值运算，当前读取模块已导入。
- `scikit-learn`、`matplotlib` 等属于后续可能使用的工具，按实际实现需要添加。

当前尚未维护固定版本的依赖清单，因此以上命令仅用于准备基础环境，不保证与已有环境的包版本完全一致。后续应记录直接依赖及验证过的版本。

## 6. 运行方法

先激活环境，再进入项目根目录运行入口。下方路径对应当前本机位置；如果移动项目，请替换成实际目录。

```powershell
conda activate pr_diabetes
Set-Location "D:\downloads\diabetes_classification"
python main.py
```

当前预期输出：

```text
项目根目录： D:\downloads\diabetes_classification
原始数据目录： D:\downloads\diabetes_classification\data\raw
输出目录： D:\downloads\diabetes_classification\outputs
```

上述输出只验证入口、包内导入和路径定位，不代表已经读取数据或完成训练。配置中的 `Path` 运算也不会自动创建目录。

统一从入口启动，不直接把包内模块作为独立脚本运行，以免相对导入缺少包上下文。

## 7. 后续开发与验证顺序

1. **读取数据**：让读取函数接收文件路径并返回 `DataFrame`，检查对象类型、行列数、列名和前几行。
2. **理解数据**：检查字段类型、缺失值、特殊字符串、重复记录及标签分布，记录观察结果。
3. **设计数据划分**：明确特征与标签；确定训练、验证和测试的使用方式，并记录随机种子。
4. **实现预处理**：根据字段含义选择处理方式；需要学习参数的步骤仅在训练数据上拟合。
5. **训练与比较模型**：在一致的评估条件下组织三种方法，不使用测试集反复调参。
6. **评估与保存结果**：除准确率外，结合任务检查混淆矩阵、精确率、召回率、F1 等指标，再分析结果。
