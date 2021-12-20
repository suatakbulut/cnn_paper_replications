import tensorflow as tf
import tensorflow.keras.layers as tfl


def alex_net(num_classes=1000, input_shape = (224,224,3)):
    """ Returns an AlexNet for a prediction model with num_classes many classes

    Notes: 
    - Here I skip the local response normalization step since unlike a 
      sigmoid activation function adopting a ReLU activation function, one will 
      not required to normalize the responses because the learning (the gradient) 
      does not depend on the value of the response.
    
    - Even though I could not see any padding before the first conv layer, for 
      the shape of the output of the first conv layer to be 55x55x96, there must 
      be a initial padding applied to the input. So, I added two rows of zeros 
      to top and left each, and one row of zeros to bottom and right each.

    Args:
        num_classes (int): num of classes to be predicted. Defaults to 1000.
        input_shape (tuple, optional): the shape of the input image. Defaults to (224,224,3).

    Returns:
        tensorflow.keras.Sequential: A Sequential AlexNet model
    """    

    input_layer = tf.keras.Input(shape=input_shape, name="input_layer")

    model = tf.keras.Sequential([
        # The first convolutional layer filters the 224x224x3 input image 
        # with 96 kernels of size 11x11x3 with a stride of 4 pixels
        tfl.ZeroPadding2D(padding=((2,1),(2,1))),
        tfl.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
        tfl.MaxPool2D(pool_size = (3,3), strides = 2),

        # The second convolutional ... filters it with 256 kernels of size 5x5x48.
        tfl.Conv2D(filters=256, kernel_size=5,  activation='relu'),
        tfl.ZeroPadding2D(padding=2),
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

    model(input_layer)

    return model 


def show_shape_at_each_layer():    
    """
    Prints the evolution of the shape of an image 
    while it passes through the network's layers
    """

    randm_img = tf.random.uniform((1,224,224,3))
    print(f"\nBelow is how the shape of a single image of shape {randm_img.shape} transforms trough each layer of the network")
    for l in alex_net().layers:
        randm_img = l(randm_img)
        if not l.__class__.__name__ == "ZeroPadding2D":
            print(f"{l.__class__.__name__:>12} output shape: {randm_img.shape}")
    print("\n")

def plot_test_train_error():
    pass 


if __name__ == "__main__":
    show_shape_at_each_layer()
    model = alex_net()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    epochs=10
    batch_size=16

    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)


