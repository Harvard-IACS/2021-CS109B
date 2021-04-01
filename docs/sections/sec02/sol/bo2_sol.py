#### Solution

class_dict = {0: "horse", 1: "zebra"}

# Define a function here to replace "sigmoid" of the final dense layer with "linear" 
# as we want the class scores, not the class

def prepare_image(img):
    img_expanded_dims = tf.expand_dims(img, axis=0)
    return img_expanded_dims

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

# Defining a function to generate saliency graphs for the 2 predicted classes
def saliency_graphs(model, img, n_classes=2, positive_gradients=False):
    """
    plot saliency maps
    
    Arguments:
        model: The trained CNN for which you want to plot gradients
        img: the input image that you want to generate saliency maps for
        n_classes (int): the number of classes in your dataset
        positive gradients (bool): if True then only plot the vanilla saliency map for the positive gradients
    """

    fig, ax = plt.subplots(1, 5, figsize=(14, 3))

    # Create Saliency object
    saliency = Saliency(model, model_modifier)
    gradcam = Gradcam(model, model_modifier)

    # pre-process img (expand dims)
    input_image = prepare_image(img)

    # predict on the input image
    y_pred = model.predict(input_image)
    col = "red" if y_pred.argmax() == 1 else "green"

    cmap_dict = {0: "Reds", 1: "Greens"}

    print("predicted class:", class_dict[y_pred.argmax()])

    # combined_saliency_map
    combined_saliency_map = np.zeros((128, 128, 3))

    for i in range(n_classes):

        # Define loss function for the class label.
        # The 'output' variable refer to the output of the model. 
        loss = lambda output: tf.keras.backend.mean(output[:, i])

        # Generate saliency map with smoothing. Smoothing reduces noise in the Saliency map
        # smooth_samples is the number of calculating gradients iterations
        saliency_map = saliency(loss, input_image, smooth_samples=20)

        # to only see positive gradients:
        if positive_gradients:
            locs = saliency_map > 0

        saliency_map = normalize(saliency_map)

        if positive_gradients:
            saliency_map[locs] = 0

        ax[i + 1].imshow(saliency_map[0, ...], cmap=cmap_dict[i])

        if i == 0:
            ax[i + 1].set_title("Horse saliency map")
        else:
            ax[i + 1].set_title("Zebra saliency map")
        combined_saliency_map[:, :, i] = saliency_map

    # show original image
    ax[0].imshow(img)

    # ploting the combined saliency map...
    # note that relative color intensity is not particularly meaningful (red vs green)
    # because we normalized the saliency maps before concatenating them into rbg.
    ax[3].imshow(img, alpha=0.3)
    ax[3].imshow(combined_saliency_map, alpha=0.96)
    ax[3].set_title("combined Saliency map")

    # gradcam for zebra class
    # we want the grad gam to be taken w.r.t. the predicted class:
    grad_cam_loss = lambda output: tf.keras.backend.mean(output[:, y_pred.argmax()])
    grad_heatmap = normalize(gradcam(grad_cam_loss, input_image))

    ax[4].imshow(img)
    ax[4].imshow(grad_heatmap[0, ...], cmap="viridis", alpha=0.6)
    ax[4].set_title("Grad Cam for pred class")

    # turn off the axis ticks for all images
    [ax[i].axis('off') for i in range(5)]
