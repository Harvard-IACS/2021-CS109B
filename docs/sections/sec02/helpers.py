# helper functions
import matplotlib.pyplot as plt

def random_rotate(image, label):
    """Dataset pipe that rotates an image, helper function to augment below"""
    shape = image.shape
    deg = tf.random.uniform([], -15., 15.)
    image = tfa.image.rotate(image, deg / 180. * np.pi, interpolation="BILINEAR")
    image.set_shape((shape))
    label.set_shape(())
    return image, label

def random_zoom(image, label):
    """Dataset pipe that zooms an image, helper function to augment below"""
    H = W = 256
    rand_float = tf.random.uniform([], 1, 10)
    rand_int = tf.cast(rand_float, tf.int32)
    image = tf.image.resize_with_crop_or_pad(image,
                                             H + H // rand_int,
                                             W + W // rand_int)
    image = tf.image.random_crop(image, size=[H, W, 3])
    return image, label


def augment(image, label):
    """Function that random augments an image via random flipping, rotation, and zoom as well as random contrast"""
    image = tf.image.random_flip_left_right(image)
    image, label = random_rotate(image, label)
    image, label = random_zoom(image, label)
    image = tf.image.random_contrast(image, lower=.25, upper=.75)

    return image, label


def plot_loss(model_history, outfile=None):
    """
    This helper function plots the NN model accuracy and loss.
    Arguments:
        model_history: the model history after fitting the NN
        out_file: the (optional) path to save the image file to.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    history = model_history
    print(history.history.keys())
    print(history.history['val_acc'][-1])

    ax[0].loglog(history.history['acc'])
    ax[0].loglog(history.history['val_acc'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'], loc='upper left')

    # summarize history for loss
    ax[1].loglog(history.history['loss'])
    ax[1].loglog(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'], loc='upper left')
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)