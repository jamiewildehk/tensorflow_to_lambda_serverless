"""
Small script to run the regression model as a standalone code for training and testing purposes
"""

import ConfigParser
import os
import numpy
import cv2
from tf_sdd_box import TensorFlowBoxingModel

# get config file
HERE = os.path.dirname(os.path.realpath(__file__))
Config = ConfigParser.ConfigParser()
Config.read(HERE + '/settings.ini')
# settings for the training
MODEL_DIR = Config.get('model', 'LOCAL_MODEL_FOLDER')
LEARNING_RATE = float(Config.get('model', 'LEARNING_RATE'))
TRAINING_EPOCHS = int(Config.get('model', 'TRAINING_EPOCHS'))
img_file = Config.get('test','IMAGE')


def main():
  img = cv2.imread(img_file,cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  r = TensorFlowBoxingModel(Config, is_training=False)
  result = r.predict(img)
  print result
  return


if __name__ == "__main__":
    main()
