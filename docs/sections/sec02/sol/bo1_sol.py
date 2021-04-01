# solutions to breakout room 1

# disable warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# expand the dimensions of the images — can also be done with .reshape(-1, *zebra_imgs[0].shape)
zebra_img_4d = np.expand_dims(zebra_imgs[0], axis=0)
horse_img_4d = np.expand_dims(horse_imgs[0], axis=0)

n_layers = 5

# solutions
for i in range(n_layers):
    # choose the model input and output
    input_layer = model.input
    output_layer = model.layers[i].output

    # modify the model
    model_ = Model(inputs=[input_layer], outputs=[output_layer])

    # make a model prediction
    zebra_layer_i_feature_maps = model_.predict(zebra_img_4d)
    horse_layer_i_feature_maps = model_.predict(horse_img_4d)

    # plot the feature maps
    quick_display(zebra_layer_i_feature_maps[0, :, :, :], title=f'zebra layer {i}')
    quick_display(horse_layer_i_feature_maps[0, :, :, :], title=f'horse layer {i}')

#re-enable warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)