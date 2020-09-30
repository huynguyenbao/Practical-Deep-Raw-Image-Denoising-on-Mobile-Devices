import tensorflow as tf
import sys
import time

def loss_function(y_ground_truth, y_pred):
  return tf.reduce_sum(tf.reduce_mean(tf.abs(y_pred - y_ground_truth), axis=0))

def train_step(model, optimizer, clean_img, noisy_img):

  variables = model.variables
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    
    tape.watch(variables)
    y_pred = model(noisy_img)
    loss = loss_function(clean_img, y_pred)

  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))

  return loss

def progress(epoch, trained_sample ,total_sample, bar_length=25, loss=0, message=""):
  percent = float(trained_sample) / float(total_sample)
  hashes = '#' * int(tf.round(percent * bar_length))
  spaces = ' ' * (bar_length - len(hashes))
  sys.stdout.write("\rEpoch {0}: [{1}] {2}%  ----- Loss: {3}".format(epoch, hashes + spaces, int(round(percent * 100)), float(loss)) + message)
  sys.stdout.flush()

def fit(model, lr, clean_imgs, noisy_imgs, epochs, checkpoints_folder):
  history = []
  optimizer = tf.keras.optimizers.Adam(lr)
  for e in range(epochs):
    start = time.time()
    total_loss = [1]
    for ite, (clean_img, noisy_img) in tf.data.Dataset.zip((clean_imgs, noisy_imgs)).enumerate():
      loss = train_step(model, optimizer, clean_img, noisy_img)
      progress(e+1, (ite+1), len(clean_imgs), loss=loss)
      total_loss.append(loss)
    history.append(sum(total_loss) / len(total_loss))
    end = time.time()
    print("\nEpoch {0}: ---------- Avg Loss: {1}, time exection: {2}".format(e+1, sum(total_loss) / len(total_loss), end - start))
    if (e+1) % 10 == 0:
      model.save_weights(checkpoints_folder)
  return history