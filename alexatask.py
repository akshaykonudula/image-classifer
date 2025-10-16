import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# 1. Load CIFAR-10 dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# For simplicity, use only 3 classes: cat (3), dog (5), horse (7)
classes_to_use = [3, 5, 7]

def filter_classes(x, y):
    idx = np.isin(y, classes_to_use).flatten()
    return x[idx], y[idx]

x_train, y_train = filter_classes(x_train, y_train)
x_test, y_test = filter_classes(x_test, y_test)

# Map labels to 0,1,2
label_map = {3:0, 5:1, 7:2}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# -------------------------------
# 2. Data augmentation
# -------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# -------------------------------
# 3. Build CNN model
# -------------------------------
model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# 4. Train model
# -------------------------------
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(x_test, y_test))

# -------------------------------
# 5. Evaluate model
# -------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Plot accuracy and loss
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("accuracy_loss.png")
print("Saved accuracy_loss.png")

# Confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Cat","Dog","Horse"])
disp.plot()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# -------------------------------
# 6. Display some predictions (Bonus)
# -------------------------------
import random
plt.figure(figsize=(10,5))
for i in range(6):
    idx = random.randint(0, len(x_test)-1)
    img = x_test[idx]
    label_true = ["Cat","Dog","Horse"][y_test[idx,0]]  # <-- fixed here
    label_pred = ["Cat","Dog","Horse"][y_pred[idx]]
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"T:{label_true} P:{label_pred}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
print("Saved sample_predictions.png")

