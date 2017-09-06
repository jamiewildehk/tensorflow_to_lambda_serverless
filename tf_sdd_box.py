# refactored from the examples at https://github.com/aymericdamien/TensorFlow-Examples
import tensorflow as tf
import numpy
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing



class TensorFlowBoxingModel:
  
  def __init__(self, config, is_training=True):
    # if it is not training, restore the model and store the session in the class
    slim = tf.contrib.slim

    net_shape = (512, 512)
    data_format = 'NHWC'
    self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    self.image_4d = tf.expand_dims(self.image_pre, 0)
    # Define the SSD model.
    reuse = True if hasattr(self, 'ssd_net') else None
    self.ssd_net = ssd_vgg_512.SSDNet()
    self.ssd_anchors = self.ssd_net.anchors(net_shape)

    with slim.arg_scope(self.ssd_net.arg_scope(data_format=data_format)):
      self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

    if not is_training:
      self.sess = self.restore_model(config.get('model', 'LOCAL_MODEL_FOLDER') + 'model.ckpt-28781')  
    return

  def restore_model(self, model):
    slim = tf.contrib.slim 
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model)
    return sess
      
  def predict(self, img, select_threshold=0.95, nms_threshold=.45, net_shape=(512, 512)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                              feed_dict={self.img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, self.ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rscores, rbboxes      
