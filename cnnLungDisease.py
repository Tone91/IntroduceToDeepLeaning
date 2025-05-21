# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

# %%
data_dir = r"C:\Users\user\Documents\PythonProjects\tf\LungDiseaseImages"
class_names = os.listdir(data_dir)  # クラス名のリスト
print(class_names)

# %%
images = []
labels = []
label_map = {name: i for i, name in enumerate(sorted(class_names))} # 辞書順にラベル付け

# 画像ファイルのパスを作成しリストに追加・ラベル付け
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.endswith('.png'):
            fpath = os.path.join(class_dir, fname)
            img = load_img(fpath, target_size=(32, 32))
            img_array = img_to_array(img) / 255.0 # 正規化
            images.append(img_array)
            labels.append(label_map[class_name])

x = np.array(images)
y = np.array(labels)
# %%
# 学習用とテスト用に分割（8:2）
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# %%
# クラスベクトルの1-hotベクトル化
y_train = to_categorical(y_train, 6)
y_test = to_categorical(y_test, 6)

# %%
print("Label map:")
for name, idx in label_map.items():
    print(f"{idx}: {name}")

# %%
# %%
model = Sequential()

# %%
# Conv2Dレイヤーを2層追加
model.add(
    Conv2D(
        filters=32, # 出力のチャンネル数（特徴マップの数）
        input_shape=(32, 32, 3),
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same', # 入力と出力のサイズを統一
        activation='relu'
    )
)

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)
# %%
# プーリング層の追加

model.add(MaxPooling2D(pool_size=(2,2)))

# %%
# ドロップアウトレイヤーの追加

model.add(Dropout(0.25))
# %%
# 畳み込み層とプーリング層の追加

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# %%
# 全結合層の追加

model.output_shape

# %%
# Flattenレイヤーの追加

model.add(Flatten())
model.output_shape

# %%
# 全結合層の追加

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=6, activation='softmax'))

# %%
# 学習

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb = TensorBoard(log_dir='./logs')
history_model1 = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)

# %%
acc = history_model1.history['accuracy']
val_acc = history_model1.history['val_accuracy']
loss = history_model1.history['loss']
val_loss = history_model1.history['val_loss']
epochs = range(1, len(acc) + 1)

# %% 2×2のサブプロットを作成
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2行2列のグラフ

# Accuracy per Epoch
axs[0, 0].plot(epochs, acc, 'bo-', label='Training Accuracy')
axs[0, 0].set_title('Accuracy per Epoch')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Loss per Epoch
axs[0, 1].plot(epochs, loss, 'ro-', label='Training Loss')
axs[0, 1].set_title('Loss per Epoch')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Validation Accuracy per Epoch
axs[1, 0].plot(epochs, val_acc, 'go-', label='Validation Accuracy')
axs[1, 0].set_title('Validation Accuracy per Epoch')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Validation Loss per Epoch
axs[1, 1].plot(epochs, val_loss, 'mo-', label='Validation Loss')
axs[1, 1].set_title('Validation Loss per Epoch')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend()
axs[1, 1].grid(True)

# グラフのレイアウト調整
plt.tight_layout()
plt.savefig('C:\\Users\\user\\Documents\\PythonProjects\\tf\\cnnLungDisease.png')
plt.show()

# %%
