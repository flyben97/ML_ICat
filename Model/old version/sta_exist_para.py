from sklearn.preprocessing import StandardScaler

def standardize_with_existing_params(data, mean, std_dev):
    """
    使用现有的均值和标准差来标准化数据。

    参数:
    data (list or numpy array): 要标准化的数据集，可以是二维数组。
    mean (list or numpy array): 每个特征的均值。
    std_dev (list or numpy array): 每个特征的标准差。

    返回:
    standardized_data (numpy array): 标准化后的数据集。
    """
    # 创建一个标准化器对象，禁用内部均值和标准差计算
    scaler = StandardScaler(with_mean=False, with_std=False)
    
    # 设置均值和标准差
    scaler.mean_ = mean
    scaler.scale_ = std_dev

    # 使用现有的均值和标准差来标准化数据
    X_standardized = scaler.transform(data)

    return X_standardized