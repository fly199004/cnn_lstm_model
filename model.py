import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Bidirectional
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Precision, Recall

# 定义批次大小和统一的向量维度
BATCH_SIZE = 8
UNIFORM_LENGTH = 512  # 假设所有词向量都填充或截断到这个长度
FEATURE_DIM = 768     # BERT基本模型的特征维度
batch_size = 8  


# 2. 修改数据加载器以同时读取特征和标签
def data_generator(file_paths, batch_size):    
    for file_path in file_paths:
        print("Loading file:", file_path)  # 调试输出
        batch_data = np.load(file_path, allow_pickle=True).item()
        features = batch_data['features']
        labels = batch_data['labels']
        # 根据批次大小将数据分块
        for i in range(0, len(features), batch_size):
            print("Loaded data shape:", features.shape, labels.shape)  # 调试输出
            yield features[i:i+batch_size], labels[i:i+batch_size]


def load_dataset(file_paths, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(file_paths, batch_size),
        output_types=(tf.float32, tf.int32),
        output_shapes=((batch_size, UNIFORM_LENGTH, FEATURE_DIM), (batch_size,))
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 划分数据集
vector_dir = 'bert_vectors'

from sklearn.model_selection import train_test_split
files = [os.path.join(vector_dir, file) for file in sorted(os.listdir(vector_dir)) if file.endswith('.npy')]
# 确保去除数据量不足的最后一个文件
sample_data = np.load(files[-1], allow_pickle=True).item()
if sample_data['features'].shape[0] < BATCH_SIZE:
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

for features, labels in train_dataset.take(1):
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)


from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization

def create_cnn_lstm_model(sequence_length, vector_dimension, num_classes):
    # 输入层
    input_layer = Input(shape=(sequence_length, vector_dimension), name="input")
    
    # 卷积和池化层
    conv_3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv_3x1")(input_layer)
    pool_3 = MaxPooling1D(pool_size=2, name="maxpool_3")(conv_3)
    conv_4 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same', name="conv_4x1")(pool_3)
    pool_4 = MaxPooling1D(pool_size=2, name="maxpool_4")(conv_4)
    conv_5 = Conv1D(filters=256, kernel_size=5, activation='relu', padding='same', name="conv_5x1")(pool_4)
    pool_5 = MaxPooling1D(pool_size=2, name="maxpool_5")(conv_5)
    
    # Batch Normalization
    batch_norm = BatchNormalization(name="batch_norm")(pool_5)
    
    # 双向LSTM层
    lstm_layer = Bidirectional(LSTM(128, return_sequences=False, name="lstm_layer"))(batch_norm)
    
    # 全连接层和Dropout
    dense_layer = Dense(128, activation='relu', name="dense_layer")(lstm_layer)
    dropout_layer = Dropout(0.5, name="dropout_layer")(dense_layer)
    
    # 输出层
    output_layer = Dense(num_classes, activation='softmax', name="output_layer")(dropout_layer)
    
    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# 模型参数
sequence_length = 512  # 序列长度
vector_dimension = 768  # 特征维度，如BERT词向量维度
num_classes = 2  # 类别数，如正面、负面

# 创建并编译模型
model = create_cnn_lstm_model(sequence_length, vector_dimension, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 打印模型概览
model.summary()

# 现在使用创建的模型进行训练
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# 这儿，需要注意，是采用的keras格式保存模型，使用H5保存会出现bug
model.save('model_3.keras')  # 使用 .keras 扩展名
print("Model saved successfully in Keras format.")

# model = tf.keras.models.load_model('model.keras')

# 使用测试集进行评估
# 假设你已经有一个适当预处理的测试数据集 test_dataset
results = model.evaluate(test_dataset)
print("Loss, Accuracy:", results)


# 初始化度量
tp = TruePositives()
tn = TrueNegatives()
fp = FalsePositives()
fn = FalseNegatives()
precision = Precision()
recall = Recall()
# 预测并更新状态
for X, y_true in test_dataset:
    y_pred = model.predict(X)
    y_pred_classes = tf.argmax(y_pred, axis=-1)  # 将输出转换为类别标签

    tp.update_state(y_true, y_pred_classes)
    tn.update_state(y_true, y_pred_classes)
    fp.update_state(y_true, y_pred_classes)
    fn.update_state(y_true, y_pred_classes)
    precision.update_state(y_true, y_pred_classes)
    recall.update_state(y_true, y_pred_classes)

# 计算 F1 score 和特异性
f1_score = 2 * (precision.result().numpy() * recall.result().numpy()) / (precision.result().numpy() + recall.result().numpy())
specificity = tn.result().numpy() / (tn.result().numpy() + fp.result().numpy())


# 输出结果
print("True Positives:", tp.result().numpy())
print("True Negatives:", tn.result().numpy())
print("False Positives:", fp.result().numpy())
print("False Negatives:", fn.result().numpy())

print("Recall:", recall.result().numpy())
print("Precision:", precision.result().numpy())
print("F1 Score:", f1_score)
print("Specificity:", specificity)
print("Loss, Accuracy:", results)