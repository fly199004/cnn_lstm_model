# 这个模型是一个典型的卷积神经网络（CNN）结构，通常用于文本分类任务。
# 下面的Python代码片段使用TensorFlow和Keras构建了这个模型。
# 假设已经有了预处理好的文本数据（词向量或嵌入矩阵）和相应的标签数据。

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, concatenate, Dense, Dropout, Flatten
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 定义批次大小和统一的向量维度
BATCH_SIZE = 8
UNIFORM_LENGTH = 512  # 假设所有词向量都填充或截断到这个长度
FEATURE_DIM = 768     # BERT基本模型的特征维度
vector_dir = 'bert_vectors_20'

# 数据加载器
# 定义生成器函数
# def data_generator(vector_dir, batch_size):
#     # 获取所有.npy文件的路径
#     vector_files = [os.path.join(vector_dir, file) for file in sorted(os.listdir(vector_dir)) if file.endswith('.npy')]
#     for file_path in vector_files:
#         data = np.load(file_path)
#         # 假设data的形状为(batch_size, sequence_length, FEATURE_DIM)，即每个.npy文件保存一个batch的数据
#         # 如果不是，可能需要调整这里的逻辑来确保每次yield一个batch
#         yield data

# # 创建TensorFlow数据集
# def load_dataset(vector_dir, batch_size):
#     # 创建生成器
#     dataset = tf.data.Dataset.from_generator(
#         lambda: data_generator(vector_dir, batch_size),
#         output_types=tf.float32,
#         output_shapes=(batch_size, None, FEATURE_DIM)  # 此处的None允许序列长度可变
#     )
#     # 将数据集扁平化，因为生成器返回的每个元素都是一个batch
#     dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
#     return dataset

# 使用load_dataset函数来创建数据集


# 这个生成器将为每个文件生成相同的数据批次，因此你可能想要在文件名中包含批次信息
# 以确保你的批次是多样化的。否则，每个批次的数据都将相同。
def data_generator(vector_dir, batch_size):
    # 获取所有.npy文件的路径
    vector_files = [os.path.join(vector_dir, file) for file in sorted(os.listdir(vector_dir)) if file.endswith('.npy')]
    for file_path in vector_files:
        data = np.load(file_path)
        # 将数据整形为 (batch_size, 512, 768) 的形状
        # 这里假设 'data' 的形状已经是 (512, 768)
        # 我们需要将其展开为 (1, 512, 768) 并沿着第一个维度重复扩展到 batch_size
        if  data is None or data.shape[0] != batch_size or data.shape[1] != UNIFORM_LENGTH or data.shape[2] != FEATURE_DIM:
            print(f"Skipping file {file_path} due to incorrect shape or None data.")
            continue  # 跳过任何形状不正确的批次
        # 如果最后一个批次数据量小于BATCH_SIZE，则填充或跳过
        remainder = data.shape[0] % batch_size
        if remainder != 0:
            # 跳过最后一个不完整的批次
            data = data[:-remainder]
        data = np.expand_dims(data, axis=0)  # 现在形状是 (1, 512, 768)
        data = np.repeat(data, batch_size, axis=0)  # 重复batch_size次，形状是 (batch_size, 512, 768)
        yield data


# 创建TensorFlow数据集
def load_dataset(vector_dir, batch_size):
    # 创建生成器
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(vector_dir, batch_size),
        output_types=tf.float32,
        output_shapes=(batch_size, UNIFORM_LENGTH, FEATURE_DIM)
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)  # 使用预提取

# 划分数据集
# 确保去除数据量不足的最后一个文件
if len(np.load(files[-1])) < BATCH_SIZE:
    files = files[:-1]

# 指定训练集、验证集和测试集的比例
train_size = 0.7
val_size = 0.15
test_size = 0.15  # Note: train_size + val_size + test_size should be 1

# 计算划分的索引
# 划分训练集、验证集、测试集文件列表
train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=val_size / (train_size + val_size), random_state=42)

# 现在你有了训练集(train_files)、验证集(val_files)和测试集(test_files)的文件列表
print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")

# 创建数据集
train_dataset = load_dataset(train_files, batch_size)
val_dataset = load_dataset(val_files, batch_size)
test_dataset = load_dataset(test_files, batch_size)

print("训练集为：",train_dataset)

def create_cnn_model(input_shape, num_classes):
    # 输入层
    input_layer = Input(shape=input_shape, name='input_layer')

    # 卷积层和池化层
    conv_3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv_3x1')(input_layer)
    pool_3 = MaxPooling1D(pool_size=2, padding='same', name='maxpool_3')(conv_3)

    conv_4 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same', name='conv_4x1')(input_layer)
    pool_4 = MaxPooling1D(pool_size=2, padding='same', name='maxpool_4')(conv_4)

    conv_5 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', name='conv_5x1')(input_layer)
    pool_5 = MaxPooling1D(pool_size=2, padding='same', name='maxpool_5')(conv_5)

    # 拼接卷积层的输出
    concatenated = concatenate([pool_3, pool_4, pool_5], axis=-1)

    # 平坦化后接一个全连接层
    flatten = Flatten()(concatenated)
    dense = Dense(128, activation='relu', name='dense_layer')(flatten)

    # Dropout层
    dropout = Dropout(0.5, name='dropout')(dense)

    # 输出层
    output_layer = Dense(num_classes, activation='softmax', name='output_layer')(dropout)

    # 创建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# 定义模型输入的维度
input_shape = (UNIFORM_LENGTH, FEATURE_DIM)  # 根据实际情况设置
num_classes = 2  # 二分类

# 调用函数创建模型
model = create_cnn_model(input_shape, num_classes)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# 现在你可以使用创建的模型进行训
# 注意，这里假设 train_dataset 是一个包含输入特征和标签的 TensorFlow 数据集对象
model.fit(train_dataset, epochs=10, validation_data=val_dataset)



