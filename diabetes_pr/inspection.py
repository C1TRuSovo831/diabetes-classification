def inspect_data(data, categorical_columns):
    print("\n===数据基础信息===")
    data.info(show_counts=True)

    print("\n===每列缺失值数量===")
    missing_counts = data.isna().sum(axis=0)
    print(missing_counts)

    for column in categorical_columns:
        print(f"\n==={column}的取值与数量===")
        counts = data[column].value_counts(dropna=False)
        print(counts)