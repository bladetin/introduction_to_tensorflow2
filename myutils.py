import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    folders = os.listdir(path_to_data)

    # 排除分类错误问题，先将目录排序
    folders = list(map(int, folders))
    folders.sort()
    folders = list(map(str, folders))

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        image_paths = glob.glob(os.path.join(full_path, '*.png'))
        x_train, x_val = train_test_split(image_paths, test_size=split_size)

        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images, path_to_csv):
    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i == 0:
                    continue
                image_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                image_full_path = os.path.join(path_to_images, image_name)
                shutil.move(image_full_path, path_to_folder)
    except:
        print('[INFO]: Error reading csv file.')


def create_generator(batch_size, train_data_path, val_data_path, test_data_path):
    train_preprocessor = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1
    )
    val_preprocessor = ImageDataGenerator(rescale=1. / 255)
    test_preprocessor = ImageDataGenerator(rescale=1. / 255)

    class_category = []
    for i in range(0, 43):
        class_category.append(i)
    class_category = list(map(str, class_category))

    # 创建训练集
    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        classes=class_category,
        target_size=(60, 60),
        color_mode='rgb',
        class_mode='categorical',
        shuffle='True',
        batch_size=batch_size
    )

    # 创建验证集
    val_generator = val_preprocessor.flow_from_directory(
        val_data_path,
        target_size=(60, 60),
        color_mode='rgb',
        class_mode='categorical',
        shuffle='False',
        batch_size=batch_size
    )

    # 创建测试集
    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        target_size=(60, 60),
        color_mode='rgb',
        class_mode='categorical',
        shuffle='False',
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator

# test
# if __name__ == '__main__':
# path_to_data = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Train'
# path_to_save_train = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\train'
# path_to_save_val = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\train_data\val'
# split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)

# path_to_images = r'C:\Users\tin\Downloads\archive\Test'
# path_to_csv = r'C:\Users\tin\Downloads\archive\Test.csv'
# order_test_set(path_to_images, path_to_csv)
