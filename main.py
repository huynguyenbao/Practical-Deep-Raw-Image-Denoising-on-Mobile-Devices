from model import DenoiseNetwork
from train import fit
from preprocess import preprocess_data
import argparse 
import os

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
    fit(model, config.lr, clean_dataset, noisy_dataset, config.epochs, config.checkpoints_folder)

