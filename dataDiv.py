# 要在Python中划分数据集为训练集、验证集和测试集，可以采用以下步骤：

# 列出所有的.npy文件。
# 随机打乱文件顺序。
# 按照给定的比例划分文件列表。
# 将这些划分后的文件列表保存下来。

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# 定义文件夹路径和批次大小
vector_dir = 'bert_vectors'
BATCH_SIZE = 8
UNIFORM_LENGTH = 512  # 假设所有词向量都填充或截断到这个长度
FEATURE_DIM = 768     # BERT基本模型的特征维度
files = [os.path.join(vector_dir, file) for file in sorted(os.listdir(vector_dir)) if file.endswith('.npy')]

# 定义数据划分函数
# def data_div():
    # 确保去除数据量不足的最后一个文件
    if len(np.load(files[-1])) < BATCH_SIZE:
        files = files[:-1]

    # 指定训练集、验证集和测试集的比例
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15  # Note: train_size + val_size + test_size should be 1

    # 计算划分的索引
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (train_size + val_size), random_state=42)

    # 现在你有了训练集(train_files)、验证集(val_files)和测试集(test_files)的文件列表
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

# 定义生成器函数
def data_generator(vector_dir, batch_size):
    # 获取所有.npy文件的路径
    vector_files = [os.path.join(vector_dir, file) for file in sorted(os.listdir(vector_dir)) if file.endswith('.npy')]
    for file_path in vector_files:
        data = np.load(file_path)
        # 假设data的形状为(batch_size, sequence_length, FEATURE_DIM)，即每个.npy文件保存一个batch的数据
        # 如果不是，可能需要调整这里的逻辑来确保每次yield一个batch
        yield data

# 创建TensorFlow数据集
def load_dataset(vector_dir, batch_size):
    data_div()
    # 创建生成器
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(vector_dir, batch_size),
        output_types=tf.float32,
        output_shapes=(batch_size, None, FEATURE_DIM)  # 此处的None允许序列长度可变
    )
    # 将数据集扁平化，因为生成器返回的每个元素都是一个batch
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    return dataset

def create_dataset(file_list, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_list),
        output_types=tf.float32,
        output_shapes=(None, UNIFORM_LENGTH, FEATURE_DIM)
    )
    # 批处理和预提取
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 之前划分的文件列表
# train_files, val_files, test_files = ...

# 创建数据集
train_dataset = create_dataset(train_files, batch_size)
val_dataset = create_dataset(val_files, batch_size)
test_dataset = create_dataset(test_files, batch_size)
