# coding: UTF-8
import os
import cv2
import csv
import sys
from PIL import Image
from keras.models import load_model
from mylib import image_util as mi
import numpy as np

def exec(names, target):
  if not resize_file(target):
    print('error ocured.')
    return
  ret = dl_eval(target)
  nameInfo = names[ret]
  print(nameInfo['id'] + "," + nameInfo['title'] + "," + nameInfo['full_name'] + "," + nameInfo['power'])

# 画像圧縮
def resize_file(target):
  if not mi.change_gray_img(target):
#    os.remove(target)
    return False
  if not mi.find_face(target):
#    os.remove(target)
    return False
  if not mi.resize_gif(target):
#    os.remove(target)
    return False
  return True

# 評価
def dl_eval(target):
  model_file_path='/home/apl/scouter/result/my_model.h5'

  # 評価
  model = load_model(model_file_path)
  image = cv2.imread(target)
  b,g,r = cv2.split(image)
  image = cv2.merge([r,g,b])
  
  X_test = []
  X_test.append(image)
  X_test=np.array(X_test)
  
  ret = model.predict(X_test)
  nameNumLabel = np.argmax(ret)
  return nameNumLabel
  
if __name__ == '__main__':
  names = []
  args = sys.argv
  list_file = os.path.join(os.path.dirname(__file__), 'charctor.list')
  with open(list_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
      names.append(row)
  exec(names, args[1])
  