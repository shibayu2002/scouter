# coding: UTF-8
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def learn(names):
  # 教師データのラベル付け
  X_train = []
  Y_train = []
  for i in range(len(names)):
    img_file_name_list=os.listdir("./data/tran/"+names[i])
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./data/tran/"+names[i]+"/",img_file_name_list[j])
        img = cv2.imread(n, cv2.IMREAD_COLOR)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)
  # テストデータのラベル付け
  X_test = []
  Y_test = []
  for i in range(len(names)):
    img_file_name_list=os.listdir("./data/test/"+names[i])
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./data/test/"+names[i]+"/",img_file_name_list[j])
        img = cv2.imread(n, cv2.IMREAD_COLOR)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        Y_test.append(i)
  X_train=np.array(X_train)
  X_test=np.array(X_test)

  from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
  from keras.models import Sequential
  from keras.utils.np_utils import to_categorical
  
  y_train = to_categorical(Y_train)
  y_test = to_categorical(Y_test)
  
  # モデルの定義
  model = Sequential()
  model.add(Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation("sigmoid"))
  model.add(Dense(128))
  model.add(Activation('sigmoid'))
  model.add(Dense(len(names)))
  model.add(Activation('softmax'))
  
  # モデル表示
  model.summary()
  
  # コンパイル
  model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
  # 学習
  history = model.fit(X_train, y_train, batch_size=8, 
                    epochs=50, verbose=1, validation_data=(X_test, y_test))
  
  # 汎化制度の評価・表示
  score = model.evaluate(X_test, y_test, batch_size=8, verbose=0)
  print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))
  
  #acc, val_accのプロット
  plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
  plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.legend(loc="best")
  plt.savefig(os.path.join(os.path.dirname(__file__), 'result/learn_result.png'))
  
  #モデルを保存
  model.save(os.path.join(os.path.dirname(__file__), 'result/my_model.h5'))
  
if __name__ == '__main__':
  names = []
  list_file = os.path.join(os.path.dirname(__file__), 'charctor.list')
  with open(list_file, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
      names.append(row['id'])
  print(names)
  learn(names)
  