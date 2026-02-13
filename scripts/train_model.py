import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train[:5000], y_train[:5000], epochs=2)
model.save("models/my_classifier_model.h5")
