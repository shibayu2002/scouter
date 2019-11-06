# coding: UTF-8
import os
import cv2
from PIL import Image

TARGET_URL = 'https://www.google.com/search?tbm=isch&q='

# グレー画像に変換
def change_gray_img(path):
  gry = cv2.imread(path, 0)
  cv2.imwrite(path, gry)
  size = os.path.getsize(path)
  if size > 0:
    return True
  else:
    return False
      
# 顔の抽出
def find_face(path):
  xml_file = os.path.join(os.path.dirname(__file__), '../conf/lbpcascade_animeface.xml')
  classifier = cv2.CascadeClassifier(xml_file)
  image = cv2.imread(path)
  print(path)
  gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#  try:
#   faces = classifier.detectMultiScale(gray_image)
# except:
#   print('Error')
#   return False
# print(len(faces))
# if len(faces) != 1:
#   return False
# print(faces)
# for i, (x,y,w,h) in enumerate(faces):
#   face_image = image[y:y+h, x:x+w]
#   cv2.imwrite(path,face_image)
#   break
# size = os.path.getsize(path)
# if size > 0:
#   return True
# else:
#   return False
  cv2.imwrite(path,gray_image)
  return True
    
# リサイズ
def resize_gif(path):
  out_width = 64
  out_height = 64
  out_img_size = out_width, out_width
  in_img = Image.open(path)
  in_img.thumbnail(out_img_size, Image.ANTIALIAS)
  blank_color = R, G, B = (255, 255, 255)
  out_img = Image.new("RGB", out_img_size, blank_color)
  in_img_width, in_img_height = in_img.size
  center = (out_width - in_img_width) // 2, (out_height - in_img_height) // 2
  out_img.paste(in_img, center)
  out_img.save(path)
  size = os.path.getsize(path)
  if size > 0:
    return True
  else:
    return False
    