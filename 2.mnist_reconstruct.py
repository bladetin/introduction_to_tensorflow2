import tensorflow as tf
import numpy as np
from deep_learning_model import functional_approach, MyCustomModel
from myutils import display_some_pic

if __name__ == '__main__':
    dataset = tf.keras.datasets.mnist.load_data()
    ((trainX, trainY), (testX, testY)) = dataset
    print('trainX.shape:', trainX.shape)
    print('trainY.shape:', trainY.shape)
    print('testX.shape:', testX.shape)
    print('testY.shape:', testY.shape)

    # if False:
    #     display_some_pic(trainX, trainY)

    # 数据预处理
    # 数据标准化[0,1]
    trainX = trainX.astype('float32') / 255
    testX = testX.astype('float32') / 255
    # 扩展维度，因为输入层多了个维度，所以需要扩展
    trainX = np.expand_dims(trainX, -1)
    testX = np.expand_dims(testX, -1)

    # model = functional_approach()
    model = MyCustomModel()

    # 模型编译
    # 三个重要参数，optimizer优化器，loss损失函数，metrics评价指标。
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    # 模型拟合数据
    # batch_size分批训练的一批有多少数量，epochs是所有图片遍历次数,validation_split是20%的图片作验证集
    model.fit(trainX, trainY, batch_size=64, epochs=3, validation_split=0.2)

    # 测试集评估
    model.evaluate(testX, testY, batch_size=64)
