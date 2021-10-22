import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

# 模型创建
# 1.tensorflow.keras.Sequential,序列模型
seq_model = keras.models.Sequential(
    [
        # 1.
        # 输入层
        Input(shape=(28, 28, 1)),
        # 2.
        # 卷积层
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        # 3.
        # 卷积层
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        # 4.
        # 输出层
        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)


# 2.funtional approach:function that returns a model，用函数定义模型，并且创建模型
def functional_approach():
    # 1.
    # 输入层
    my_input = Input(shape=(28, 28, 1))
    # 2.
    # 卷积层
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # 3.
    # 卷积层
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # 4.
    # 输出层
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.models.Model(inputs=my_input, outputs=x)

    return model


# 3.tensorflow.keras.Model:inherit from this class
class MyCustomModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='relu')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def display_some_pic(examples, labels):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        index = np.random.randint(0, len(examples) - 1)
        img = examples[index]
        label = labels[index]

        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')

    plt.show()


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
