import os.path

import numpy as np
import tensorflow as tf


def predict_with_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60, 60])  # (60,60,3)
    image = tf.expand_dims(image, axis=0)  # (1,60,60,3)

    predictions = model.predict(image)  # 返回一个列表，列表大小是种类个数，[0.001,0.03,0.94,......]
    predictions = np.argmax(predictions)  # 返回最大的索引，即2

    return predictions


if __name__ == '__main__':
    # img_path = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Test\2\10025.png'
    base_path = r'E:\Tin\其他文件\Desktop\数据集\German Traffic Sign Recognition Benchmark\Test'
    model = tf.keras.models.load_model(r'.\Models')
    image_path = []
    num_error = 0

    for i in range(0, 43):
        class_path = os.path.join(base_path, str(i))
        all_class_path = os.listdir(class_path)
        image_path.append(
            os.path.join(class_path, all_class_path[np.random.randint(0, len(os.listdir(class_path)) - 1)]))
        print(f'actual class: {i} ')
        prediction = predict_with_model(model, image_path[i])
        print(f'prediction: {prediction}')

        if i != prediction:
            num_error = num_error + 1

        print(num_error / 43.)
