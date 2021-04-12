# Crucially, observe that we sum across the image dimensions and only take the mean in the 
# images dimension
def reconstruction_loss(y_target, y_predicted):
    loss = tf.keras.losses.mean_squared_error(y_target, y_predicted)
    # tf.reduce_sum is over all pixels and tf.reduce_mean is over all images
    return tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2])) 


def make_ConvAE(activation, 
input_dim, embed_dim):
    """
    A convolutional auto-encoder encoder for an AE.
    
    Parameters
    ----------
        activation: string
            the activation function to use in the encoding
        input_shape: int tuple ex (128, 128, 3)
            the input shape of the image
        embed_dim:
        
    """
    act_down = {"activation" : activation, "strides": (2,2), 'padding' : 'SAME' }
    act_up = {"activation" : 'relu',
              'kernel_size': (4,4),
              "strides": (1,1),
              'padding' : 'SAME' }
    upsamp_args = {'size' : (2,2), 'interpolation' : 'nearest'}
    flat_embed_dim = (np.prod(embed_dim),)
    encoder = Sequential([
                Conv2D(2**4, kernel_size = (3,3), input_shape = input_dim, **act_down),
                Conv2D(2**5, kernel_size = (2,2), input_shape = input_dim, **act_down),
                Conv2D(2**5, kernel_size = (2,2), **act_down),
                Dropout(0.05),
                Flatten(),
                
                Dense(flat_embed_dim[0], activation = "linear"), #begin embedding space
                Dense(flat_embed_dim[0], activation = "relu"), #end embedding space
                
                layers.Reshape(embed_dim)])
    decoder = Sequential([
                Input(shape = embed_dim),
                
                layers.UpSampling2D(**upsamp_args),
                Conv2D(2**5, **act_up),
                layers.UpSampling2D(**upsamp_args),
                Conv2D(2**4, **act_up),
                layers.UpSampling2D(**upsamp_args),
                Conv2D(2**3, **act_up),
                layers.UpSampling2D(**upsamp_args), 
                Conv2D(2**2, **act_up),
                layers.UpSampling2D(**upsamp_args), 
                Conv2D(3, activation = "sigmoid",
                             kernel_size = (1,1),
                             strides = (1,1),
                             padding = "SAME"),
                ])
    _input = Input(shape = INPUT_SIZE)
    output = decoder(encoder(_input))
    conv_AE = models.Model(inputs = _input, outputs = output)
    conv_AE.compile(optimizer = Adam(learning_rate=1e-4), loss = reconstruction_loss)
    conv_AE.summary()
    
    return conv_AE
