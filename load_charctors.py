# coding: UTF-8
import urllib3
import urllib
from bs4 import BeautifulSoup
import certifi
import requests
import json
import sys
import uuid
import os
import csv
import cv2
import numpy as np
import random
import shutil
from mylib import image_util as mi

TARGET_URL = 'https://www.google.com/search?tbm=isch&q='

# アニメ画像検索
def findAnimeGif(data_dir, test_dir, keyword):
  print('find ' + keyword)
  
  session = requests.session()
  session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) \
    Gecko/20100101 Firefox/10.0"
  })
  http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
  
  count = 0;
  for i in range(3):
    url = TARGET_URL + urllib.parse.quote(keyword) + "&ijn=" + str(i)
    print(url)
    html = session.get(url).text
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
    elements = soup.select(".rg_meta.notranslate")
    jsons = [json.loads(e.get_text()) for e in elements]
    for js in jsons:
        print(js['ou'])
        ret = download_file(js['ou'], data_dir, test_dir)
        if ret:
          count = count + 1
          if count >= 50 :
            break

# 画像ダウンロード
def download_file(url, data_path, test_dir):
  if random.random() < 0.1:
    dst_path = test_dir
  else:
    dst_path = data_path
  try:
    r = requests.get(url, timeout=(3.0, 20.0))
  except:
    print('Error')
    return False
  path = dst_path + str(uuid.uuid4()) + str('.jpg')
  with open(path, 'wb') as file:
    file.write(r.content)
    size = os.path.getsize(path)
    if size <= 0:
      os.remove(path)
      return False
    if not mi.change_gray_img(path):
      os.remove(path)
      return False
    if not mi.find_face(path):
      os.remove(path)
      return False
    if not mi.resize_gif(path):
      os.remove(path)
      return False
    # 画像の水増し
    image = cv2.imread(path)
    scratch_face_images = scratch_image(image)
    # 画像の保存
    for idx, image in enumerate(scratch_face_images):
      if random.random() < 0.1:
        dst_path = test_dir
      else:
        dst_path = data_path
      output_path = dst_path + str(uuid.uuid4()) + str('.jpg')
      cv2.imwrite(output_path, image)
    return True

# 画像水増し処理
def scratch_image(image, use_flip=True, use_threshold=True, use_filter=True):
    # どの水増手法を利用するか（フリップ、閾値、平滑化）
    methods = [use_flip, use_threshold, use_filter]
    # ぼかしに使うフィルターの作成
    # filter1 = np.ones((3, 3))
    # オリジナルの画像を配列に格納
    images = [image]
    # 水増手法の関数
    scratch = np.array([
        # フリップ処理
        lambda x: cv2.flip(x, 1),
        # 閾値処理
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        # 平滑化処理
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 画像の水増
    doubling_images = lambda f, img: np.r_[img, [f(i) for i in img]]
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images
    
if __name__ == '__main__':
  list_file = os.path.join(os.path.dirname(__file__), 'charctor.list')
  with open(list_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
      print(row)
      data_dir = os.path.join(os.path.dirname(__file__), 'data/tran/' + row['id'] + '/')
      test_dir = os.path.join(os.path.dirname(__file__), 'data/test/' + row['id'] + '/')
      os.makedirs(data_dir, exist_ok=True)
      os.makedirs(test_dir, exist_ok=True)
      shutil.rmtree(data_dir)
      shutil.rmtree(test_dir)
      os.makedirs(data_dir, exist_ok=True)
      os.makedirs(test_dir, exist_ok=True)
      findAnimeGif(data_dir, test_dir, row['title'] + " " + row['name'] + ' 顔')
