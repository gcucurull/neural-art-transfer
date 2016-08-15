import numpy as np
import tensorflow as tf

def content_loss(cont_out, target_out, layer, content_weight):
    '''
        # content loss is just the mean square error between the outputs of a given layer
        # in the content image and the target image
    '''
    cont_loss = tf.reduce_sum(tf.square(tf.sub(target_out[layer], cont_out)))

    # multiply the loss by it's weight
    cont_loss = tf.mul(cont_loss, content_weight, name="cont_loss")
    #tf.add_to_collection('losses', cont_loss)

    return cont_loss

def get_shape(inp):
    # returns the shape of a tensor or an array
    if type(inp) == type(np.array([])):
        return inp.shape
    else:
        return [i.value for i in inp.get_shape()]

def style_loss(style_out, target_out, layers, style_weight_layer):

    def style_layer_loss(style_out, target_out, layer):
        '''
            # returns the style loss for a given layer between
            # the style image and the target image
        '''
        def gram_matrix(activation):
            flat = tf.reshape(activation, [-1, get_shape(activation)[3]]) # shape[3] is the number of feature maps
            res = tf.matmul(flat, flat, transpose_a=True)
            return res

        N = get_shape(target_out[layer])[3] # number of feature maps
        M = get_shape(target_out[layer])[1] * get_shape(target_out[layer])[2] # dimension of each feature map
        
        # compute the gram matrices of the activations of the given layer
        style_gram = gram_matrix(style_out[layer])
        target_gram = gram_matrix(target_out[layer])

        st_loss = tf.mul(tf.reduce_sum(tf.square(tf.sub(target_gram, style_gram))), 1./((N**2) * (M**2)))

        # multiply the loss by it's weight
        st_loss = tf.mul(st_loss, style_weight_layer, name='style_loss')

        #tf.add_to_collection('losses', st_loss)
        return st_loss

    losses = []
    for s_l in layers:
        loss = style_layer_loss(style_out, target_out, s_l)
        losses.append(loss)

    return losses

def total_var_loss(generated, tv_weight):
    ''' 
        Computes the total variation loss of the generated image
    '''
    batch, width, height, channels = get_shape(generated)

    width_var = tf.nn.l2_loss(tf.sub(generated[:,:width-1,:,:], generated[:,1:,:,:]))
    height_var = tf.nn.l2_loss(tf.sub(generated[:,:,:height-1,:], generated[:,:,1:,:]))

    return tv_weight*tf.add(width_var, height_var)