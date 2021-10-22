import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten


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


# 交通指示牌训练模型
def streetsigns_model(num_classes):
    # 1.输入层
    my_input = Input(shape=(60, 60, 3))
    # 2.卷积层
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # 3.卷积层
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # 4.卷积层
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    # 5.输出层
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = keras.models.Model(inputs=my_input, outputs=x)

    return model


# 测试代码
if __name__ == '__main__':
    test_model = streetsigns_model(10)
    test_model.summary()
