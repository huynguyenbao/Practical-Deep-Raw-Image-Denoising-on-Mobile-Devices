from model import DenoiseNetwork
import tensorflow as tf
from preprocess import preprocess_data
import argparse 
import os

def loss_function(y_ground_truth, y_pred):
  return tf.reduce_sum(tf.reduce_mean(tf.abs(y_pred - y_ground_truth), axis=0))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoints_folder', type=str, default="weights//ckpt")
    parser.add_argument('--pretrain_dir', type=str)

    config = parser.parse_args()

    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    # Get Data
    clean_dataset, noisy_dataset = preprocess_data()

    # Create Model
    model = DenoiseNetwork()
    model.build((None,512,512,3))
    if config.pretrain_dir:
        model.load_weights(config.pretrain_dir)

    # Train Model
    opt = tf.keras.optimizers.Adam(config.lr)
    model.compile(optimizer=opt, loss=loss_function, metrics=[loss_function])
    model.fit(tf.data.Dataset.zip((noisy_dataset, clean_dataset)), epochs=config.epochs)