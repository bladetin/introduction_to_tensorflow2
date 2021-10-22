import tensorflow.keras as keras
from myutils import split_data, order_test_set, create_generator
from deep_learning_model import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    # 处理训练集和验证集
    # path_to_data = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Train'
    # path_to_save_train = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\train'
    # path_to_save_val = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\val'
    # split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

    # 处理测试集
    # path_to_images = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Test'
    # path_to_csv = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Test.csv'
    # order_test_set(path_to_images, path_to_csv)

    path_to_train = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\train'
    path_to_val = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\val'
    path_to_test = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Test'
    batch_size = 64
    epochs = 15
    lr = 0.0001

    train_generator, val_generator, test_generator = create_generator(batch_size, path_to_train, path_to_val,
                                                                      path_to_test)
    num_classes = train_generator.num_classes

    # 训练模型还是测试模型
    TRAIN = True
    TEST = True

    if TRAIN:
        # callbacks，回调函数系列,保存模型
        path_to_model = r'.\Models'
        checkpoint_saver = ModelCheckpoint(
            path_to_model,
            monitor='val_accuracy',
            mode='max',
            save_best_only='True',
            save_freq='epoch',
            verbose=1
        )
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10)  # 提前退出机制：10个epoch没有进步的话，就退出

        model = streetsigns_model(num_classes)
        my_optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[checkpoint_saver, early_stop]
        )

    if TEST:
        model = keras.models.load_model(r'.\Models')
        model.summary()

        print('Evaluate validation set: ')
        model.evaluate(val_generator)

        print('Evaluate test set: ')
        model.evaluate(test_generator)
