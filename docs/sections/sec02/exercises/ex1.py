# exercise: try to print out multiple maps using the quick_display function

# disable warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

zebra_img = zebra_imgs[0]
horse_img = horse_imgs[0]

# expand the dimensions of the images of zebra_img and horse_img (can also be done with reshape)
zebra_img_4d = np.expand_dims(..., axis=...)
horse_img_4d = np.expand_dims(..., axis=...)

n_layers = 3

# solutions
for i in range(n_layers):
    # choose the model input and output
    input_layer = ...
    output_layer = ...

    # modify the model
    model_ = Model(inputs=[input_layer], outputs=[output_layer])

    # make a model prediction
    zebra_layer_i_feature_maps = model_.predict(...)
    horse_layer_i_feature_maps = model_.predict(...)

    # plot the feature maps (don't forget to convert them back to 3d before plotting!)
    quick_display(..., title=f'zebra layer {i}')
    quick_display(..., title=f'horse layer {i}')

#re-enable warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)