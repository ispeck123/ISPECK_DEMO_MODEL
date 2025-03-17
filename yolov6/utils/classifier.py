import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
test_barricade_model = load_model('F:/heavy_machinary_yolo6/YOLOv6/updated_cover_barricade_model.h5')
class Inference:
  def __init__(self,im,resize=(224,224),kernel_size=5):
    self.im = im
    self.resize = resize
    self.kernel_size = kernel_size

  def predicted(self):
    resized_im = cv2.resize(self.im,self.resize)
    filtered_im = cv2.medianBlur(resized_im,self.kernel_size)
    normalized_im = filtered_im /255.0
    tensor_im = tf.convert_to_tensor(normalized_im)
    predict = test_barricade_model(tf.expand_dims(tensor_im,axis=0))
    predicted_value = tf.math.argmax(predict , axis = -1)
    if predicted_value.numpy()[0] == 0:
      return 'barricade' ,round(predict.numpy()[0][0],2)
    elif predicted_value.numpy()[0] == 1:
      return 'missing_barricade',round(predict.numpy()[0][1],2)