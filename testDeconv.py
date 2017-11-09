import tensorflow as tf
import numpy as np

height = 10
width = 10

def conv2d_stride2(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape,n):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=n)


def bias_variable(shape,n):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial,name=n)

def forward(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, height, width, 3])

    #height/2 * width/2
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 3, 64],'w1')
        b_conv1 = bias_variable([64],'b1')
        h_conv1 = tf.nn.relu(conv2d_stride2(x_image, W_conv1) + b_conv1)

    #height/4 * width/4
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 128],'w2')
        b_conv2 = bias_variable([128],'b2')
        h_conv2 = tf.nn.relu(conv2d_stride2(h_conv1, W_conv2) + b_conv2)

    #deconv: height * width * 3 
    with tf.name_scope('deconv'):
        W_deconv = weight_variable([5, 5, 3, 128],'wd')
        deconv = tf.nn.conv2d_transpose(h_conv2, W_deconv, [1, 10, 10, 3], [1, 4, 4, 1], padding='SAME', name=None)
    return deconv

if __name__ == '__main__':
    sess = tf.Session()
    x = tf.placeholder(tf.float32, [None, height, width, 3])
    dec = forward(x)
    xtest = np.random.random([1,10,10,3])
    sess.run(tf.global_variables_initializer())
    y = sess.run(dec,feed_dict={x:xtest})
    print(y.shape)