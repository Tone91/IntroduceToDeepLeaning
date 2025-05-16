# %%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyolot as plt

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
print('x_train.shape :', x_train.shape)
print('x_test.shape :', x_test.shape)
print('y_train.shape :', y_train.shape)
print('y_test.shape :', y_test.shape)

# %%
# 特徴量の正規化
x_train = x_train / 255
x_test = x_test / 255

# クラスベクトルの1-hotベクトル化
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

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
model.add(Dense(units=10, activation='softmax'))

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
plt.savefig('C:\\Users\\user\\Documents\\PythonProjects\\tf\\cnn.png')
plt.show()
