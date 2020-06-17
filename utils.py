import math
import numpy as np
import tensorflow as tf

def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (150, 150), method='gaussian', preserve_aspect_ratio=False, antialias=True)
  image /= 255
  return image, label
