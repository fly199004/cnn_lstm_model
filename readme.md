## 主题：基于CNN-LSTM的客户评价文本情感分析实验

整个内容是根据“季季红”的客户评价信息来做基于CNN-LSTM的情感分析，所有实验及代码文件说明 



![image-20240423220921583](https://site-1314099117.cos.ap-guangzhou.myqcloud.com/img/image-20240423220921583.png)

## 数据文件说明

- jijihong.csv ，是经过初期处理过的文件 ，里面仅包含评价文本“text”和“tags”两列信息，有1万行信息 
- data_jieba.csv， jieba文本分词后的数据文件
- data_syn.csv，同义转化文本处理后的数据
- labels_1.csv，提取出的标签数据
- sentiment_data.csv，文本标记后的结果
- data_syn_with_label.csv ，经过3.3节整个文本预处理之后，并且加上标签的数据，约8千多数据
- data_syn_with_label_200.csv，取前200条数据来完善模型功能



## 生成数据文件

- bert_vectors目录，保存了词向量表征后的数据，npy文件，每个文件的batch_size = 8，约有12G大小，约1千个文件

- bert_vectors_200目录，用前200条评价数据来测试模型是否能正常运行 

- csv文件，原始数据，数据处理过程中产生的新数据

  

## python文件

- model.py文件，导入数据，划分数据，创建LSTM模型，然后训练LSTM模型，从数据导入到评估模型性能一整套流程，只需要在代码里面直接更换各个模型代码，便能直接运行出各个模型的评估参数
- 
- 

## jupiter notebook文件

- autodraw.ipynb, 绘制框架图、模型图
- dataDiv.ipynb , 数据划分 
- dataProcess.ipynb ，数据清理、筛选、构建同义词曲
- labelProcess.ipynb，提取标签
- bert.ipynb，bert词向量表征
- bert_with_label.ipynb，加上标签后进行词向量表征 
- lert_with_label_with_CNN.ipynb，词向量表征后，数据划分，然后运行出CNN模型
- lstm_model.ipynb，导入数据，划分数据，创建LSTM模型，然后训练LSTM模型
- cnn_lstm.ipynb， 训练CNN-LSTM模型
- cnnModel.ipynb,训练CNN模型
- svm_model.ipynb，训练SVM模型
- test_score.ipynb, 加载保存好的模型文件，加载数据，然后评估模型评估参数
- cnn_lstm_model-V.ipynb，成功运行之后，更换超参数设置
- bert_visual.ipynb, bert向量表征化的可视化效果
- draw.ipynb， 绘制了实验结果各种折线图、雷达图等

## 图片文件

中途要设计各种网络结构，通过代码设计生成的架构图片



## 模型文件

- trained_cnn_model.h5 ，是经过训练后的CNN模型，已经训练了完全的数据。

  - **文件格式**：`.h5` 文件是一个HDF5文件，它是一个存储大量数值数据的文件格式，TensorFlow 和 Keras 使用这种格式来保存模型的结构以及权重（参数）。
  - **持久化**：一旦保存了模型，它就会持久存在于磁盘上，即使关闭了代码运行的环境（如关闭了你的编程环境或电脑），这个文件仍然存在。你可以随时加载这个模型进行预测、继续训练或进一步分析。
  - **加载模型**：使用 `tf.keras.models.load_model('trained_model.h5')` 可以加载之前保存的模型。这个功能让你可以在不同的时间、不同的环境中重新加载并使用该模型，而不需要重新训练。
  - **继续使用**：加载模型后，你可以直接使用它来做预测、评估或者继续训练。这是机器学习项目中常见的需求，因为训练模型通常需要大量的计算资源和时间，一旦训练完成，能够保存并重新使用训练好的模型是非常有价值的。
- train_ltsm_model.h5 ,LSTM模型
- train_cnn_lstm_model.h5 , CNN-LSTM模型
- train_cnn_lstm_model.keras , CNN-LSTM模型，后改用这个格式的模型。数据加载不易出错
- model.keras ,后续调整过程中保存的模型
- model_2.keras
- model_3.keras

## 其他文件

- stopwords_hit.txt ，哈工大停用词表

- 思维导图，实验想法和流程梳理

  

![image-20240423221049795](D:\Code\Python\dataProcess\image-20240423221049795.png)
