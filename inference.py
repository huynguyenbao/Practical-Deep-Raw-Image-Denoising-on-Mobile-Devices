from model import DenoiseNetwork
from preprocess import load_image

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--pretrain_dir', type=str, default= "weights//ckpt")

	config = parser.parse_args()

    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    # Load model
    model = DenoiseNetwork()
    model.build((None,512,512,3))
    model.load_weights(config.pretrain_dir)

    # Inference
    noisy_image = load_image(input_path)    
    clean_image = model(noisy_image[None])
    
    # Save denoised image
    tf.keras.preprocessing.image.save_img(output_path, clean_image[0])