# %%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# %%
# データのインポート
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

# %%
x_train = x_train.reshape(60000, 784)
x_train = x_train/255.
x_test = x_test.reshape(10000, 784)
x_test = x_test/255.

# %%
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

# %%
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %%
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

# %%
# ネットワークの構築

model = Sequential()

# %%
# 中間層の追加
model.add(
    Dense(
        units=64,   # ニューロンの数
        input_shape=(784,), # 入力されるテンソルの数
        activation='relu'   # 活性化関数の種類
    )
)

# %%
# 出力層の追加
model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

# %%
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir='logs')
history_adam=model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)

# %%
acc = history_adam.history['accuracy']
val_acc = history_adam.history['val_accuracy']
loss = history_adam.history['loss']
val_loss = history_adam.history['val_loss']
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
plt.savefig('C:\\Users\\user\\Documents\\PythonProjects\\tf\\ffnnSequential.png')
plt.show()