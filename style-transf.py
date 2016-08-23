import scipy.misc
import numpy as np
import tensorflow as tf
from models import alexnet
from models import vgg
import argparse
import losses

network_model = vgg

input_content = 'input/1-content.jpg'
input_style = 'styles/1-style.jpg'

def load_image(img_path, new_size=None):
    loaded = scipy.misc.imread(img_path).astype(np.float)

    # grayscale to rgb
    if len(loaded.shape) == 2:
        loaded = np.dstack([loaded, loaded, loaded])

    # rescale if needed
    if new_size:
        h,w,c = loaded.shape
        if h > w:
            ratio = w/float(h)
            shape = (new_size, int(new_size*ratio))
        else:
            ratio = h/float(w)
            shape = (int(new_size*ratio), new_size)
        loaded = scipy.misc.imresize(loaded, shape)

    return loaded

def get_name(photo):
    return photo.split('/')[1].split('.')[0]

C_LAYER = network_model.content_layers()
S_LAYERS = network_model.style_layers()

content_weight = 1e0
style_weight = 1e3 #1e2
tv_weight = 0
learning_rate = 1e0

ITERATIONS = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--iter',
        dest='iter', help='number of iteraions', 
        default=ITERATIONS, type=int)
parser.add_argument('--cont',
        dest='cont', help='content image', 
        default=input_content)
parser.add_argument('--style',
        dest='style', help='style image', 
        default=input_style)
parser.add_argument('--out',
        dest='out', help='output name')
parser.add_argument('--lr',
        dest='learning_rate', help='learning rate', 
        default=learning_rate, type=float)
parser.add_argument('--cont_w',
        dest='content_weight', help='content weight', 
        default=content_weight, type=float)
parser.add_argument('--style_w',
        dest='style_weight', help='style weight', 
        default=style_weight, type=float)
parser.add_argument('--tv_w',
        dest='tv_weight', help='tv weight', 
        default=tv_weight, type=float)
parser.add_argument('--cont_size',
        dest='cont_size', help="Size of the largest dimension for the content image",
        default=None, type=int)
parser.add_argument('--style_size',
        dest='style_size', help="Size of the largest dimension for style image",
        default=None, type=int)

options = parser.parse_args()

print(options)

if not options.out:
    out = get_name(options.cont)+'_'+get_name(options.style)+'.jpg'
else:
    out = options.out

style_weight_layer = options.style_weight/len(S_LAYERS)

content = load_image(options.cont, options.cont_size)
style = load_image(options.style, options.style_size)

# compute layer activations for content
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    content_pre = np.array([network_model.preprocess(content)])

    image = tf.placeholder('float', shape=content_pre.shape)
    model = network_model.get_model(image)
    content_out = sess.run(model[C_LAYER], feed_dict = {image:content_pre})

# compute layer activations for style
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    style_pre = np.array([network_model.preprocess(style)])
    image = tf.placeholder('float', shape=style_pre.shape)
    model = network_model.get_model(image)
    style_out = sess.run({s_l:model[s_l] for s_l in S_LAYERS}, feed_dict = {image:style_pre})

# create image merging content and style
g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    # init randomly
    # white noise
    target = tf.random_normal((1,)+content.shape)

    target_pre_var = tf.Variable(target)

    # build model with empty layer activations for generated target image
    model = network_model.get_model(target_pre_var)

    # compute loss
    cont_cost = losses.content_loss(content_out, model, C_LAYER, content_weight)
    style_cost = losses.style_loss(style_out, model, S_LAYERS, style_weight_layer)
    tv_cost = losses.total_var_loss(target_pre_var, tv_weight)

    total_loss = cont_cost + tf.add_n(style_cost) + tv_cost
    # total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    sess.run(tf.initialize_all_variables())
    min_loss = float("inf")
    best = None
    for i in range(options.iter):
        train_step.run()
        print('Iteration %d/%d' % (i + 1, options.iter))

        if (i%5 == 0):
            loss = total_loss.eval()
            print('    total loss: %g' % total_loss.eval())
            if(loss < min_loss):
                min_loss = loss
                best = target_pre_var.eval()

    print('  content loss: %g' % cont_cost.eval())
    print('    style loss: %g' % tf.add_n(style_cost).eval())
    print('       tv loss: %g' % tv_cost.eval())
    print('    total loss: %g' % total_loss.eval())


    final = best
    final = final.squeeze()
    final = network_model.postprocess(final)

    final = np.clip(final, 0, 255).astype(np.uint8)

    scipy.misc.imsave(out, final)
