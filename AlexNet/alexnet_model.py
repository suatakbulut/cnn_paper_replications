import tensorflow as tf
import tensorflow.keras.layers as tfl
# import tensorflow_datasets as tfds

def nnet(num_classes):
    model = tf.keras.Sequential([
        # The first convolutional layer filters the 224 224 3 input image 
        # with 96 kernels of size 11x11x3 with a stride of 4 pixels
        tfl.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
        tfl.MaxPool2D(pool_size = (3,3), strides = 2),

        # The second convolutional ... filters it with 256 kernels of size 5x5x48.
        tfl.Conv2D(filters=256, kernel_size=5, padding = "same", activation='relu'),
        tfl.MaxPool2D(pool_size = (3,3), strides = 2),
        
        # The third, fourth, and fifth convolutional layers are connected to 
        # one another without any intervening pooling or normalization layers.

        # The third convolutional layer has 384 kernels of size 3x3x256
        tfl.Conv2D(filters=384, kernel_size=3, padding = "same", activation='relu'),      
        # The fourth convolutional layer has 384 kernels of size 3x3x192
        tfl.Conv2D(filters=384, kernel_size=3, padding = "same", activation='relu'),
        # and the fifth convolutional layer has 256 kernels of size 3x3x192
        tfl.Conv2D(filters=256, kernel_size=3, padding = "same", activation='relu'),
        tfl.MaxPool2D(pool_size = (3,3), strides = 2),

        tfl.Flatten(), 
        # The fully-connected layers have 4096 neurons each.
        tfl.Dense(4096, activation = "relu"), 
        tfl.Dropout(0.5),

        tfl.Dense(4096, activation = "relu"), 
        tfl.Dropout(0.5),

        tfl.Dense(num_classes) ])
    return model 


def show_shape_at_each_layer(num_classes):
    randm_img = tf.random.uniform((1,224,224,1))
    print(f"\nBelow is how the shape of a single image of shape {randm_img.shape[1:]} transforms trough each layer of the network")
    for l in nnet(num_classes).layers:
        randm_img = l(randm_img)
        print(f"{l.__class__.__name__:>12} output shape: {randm_img.shape[1:]}")
    print("\n")


if __name__ == "__main__":
    num_classes = 10 
    show_shape_at_each_layer(num_classes)



"""
batch_size = 16
num_classes = 10 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=16)
model.evaluate(X_test, Y_test)
"""
