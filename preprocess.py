import tensorflow as tf

def load_image(path, img_size=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, 3)
    img = tf.cast(img, tf.float32)
    
    if img_size:
        img = tf.image.resize(img, img_size)

    return img / 255.0


def preprocess_data():
    BATCH_SIZE = 16

    data_path1 = "/content/drive/My Drive/Data/sidd_rgb1/"
    data_path2 = "/content/drive/My Drive/Data/sidd_rgb2/"

    BATCH_SIZE = 8

    clean_path1 = tf.data.Dataset.list_files(data_path1 + '/clean/*.png', shuffle=False)
    noisy_path1 = tf.data.Dataset.list_files(data_path1 + '/noisy/*.png', shuffle=False)

    clean_path2 = tf.data.Dataset.list_files(data_path2 + '/clean/*.png', shuffle=False)
    noisy_path2 = tf.data.Dataset.list_files(data_path2 + '/noisy/*.png', shuffle=False)

    clean_path = clean_path1.concatenate(clean_path2)
    noisy_path = noisy_path1.concatenate(noisy_path2)

    clean_dataset = clean_path.map(lambda x: load_image(x)).batch(BATCH_SIZE)
    noisy_dataset = noisy_path.map(lambda x: load_image(x)).batch(BATCH_SIZE)

    return clean_dataset, noisy_dataset