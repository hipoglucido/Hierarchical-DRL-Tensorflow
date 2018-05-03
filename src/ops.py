import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def clipped_error(y_true, y_pred):
    x = tf.abs(y_true - y_pred)
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def linear(input_, output_size, stddev=0.0002, bias_start=0.0,
    activation_fn = None, name = 'linear'):
    
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.contrib.layers.xavier_initializer(uniform = True))
        b = tf.get_variable('bias', [output_size], 
                initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)
#        out = tf.matmul(input_, w)
    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b

"""Loss functions."""

import tensorflow as tf


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.
    See https://en.wikipedia.org/wiki/Huber_loss
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(a)
    greater_than_max = max_grad * (a - 0.5 * max_grad)
    return tf.where(a <= max_grad, x=less_than_max, y=greater_than_max)



def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad=max_grad))


def weighted_huber_loss(y_true, y_pred, weights, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    weights: np.array, tf.Tensor
      weights value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(weights*huber_loss(y_true, y_pred, max_grad=max_grad))