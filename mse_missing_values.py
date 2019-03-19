from keras import backend as K
import tensorflow as tf


def missing_mse(y_true, y_pred):
    """ test missing values defined as -1.0. Loss should be 1.0
    >>>import tensorflow as tf
    >>>from keras import backend as K
    >>>sess = tf.session()
    >>>y_true_test = tf.constant([[-1.0,2.0],[5.0,-1.0]])
    >>>y_pred_test = tf.constant([[2.0,3.0],[4.0,2.0]])
    >>>float_missing_test = K.cast(tf.logical_not(tf.equal(y_pred,-1.0)), dtype='float32')
    >>>tf_float_missing = sess.run(float_missing_test)
    >>>tf_float_missing
    array([[0., 1.],
       [1., 0.]], dtype=float32)
    >>>loss_test = K.mean(K.square( (y_pred - y_true)*float_missing ), axis=-1)
    >>>sess.run(loss_test)
    array([0.5, 0.5], dtype=float32)
    >>>assert (sess.run(missing_mse(y_true_test, y_pred_test)) == sess.run(K.sum(loss_test)))
    """
    # y_pred=tf.constant([[1.0,2.0],[5.0,10.0]])

    float_missing = K.cast(tf.logical_not(tf.equal(y_true, -1.0)), dtype='float32')
    loss_vector =  K.mean(K.square( (y_pred - y_true)*float_missing ), axis=-1)
    return K.sum(loss_vector)

def missing_mse2(y_true, y_pred):

    """ test missing values defined as -1.0. Loss should be 1.0
    >>>import tensorflow as tf
    >>>from keras import backend as K
    >>>sess = tf.session()
    >>>y_true_test = tf.constant([[-1.0,2.0],[5.0,-1.0]])
    >>>y_pred_test = tf.constant([[2.0,4.0],[3.0,2.0]])
    >>> sess.run(y_true_test)
    array([[-1.,  2.],
           [ 5., -1.]], dtype=float32)
    >>> sess.run(y_pred_test)
    array([[2., 4.],
           [3., 2.]], dtype=float32)
    >>> bool_missing = tf.logical_not(tf.equal(y_true_test,-1.0))
    >>> sess.run(bool_missing)
    array([[False,  True],
           [ True, False]])
    >>> float_missing = K.cast(bool_missing, dtype='float32')
    >>> sess.run(float_missing)
    array([[0., 1.],
           [1., 0.]], dtype=float32)
    >>> missing_squared = K.square( (y_pred_test - y_true_test)*float_missing )
    >>> sess.run(missing_squared)
    array([[0., 4.],
           [4., 0.]], dtype=float32)
    >>> masked = tf.boolean_mask(missing_squared, bool_missing)
    >>> sess.run(masked)
    array([4., 4.], dtype=float32)
    >>> loss_vector = K.mean(masked, axis=-1)
    >>> sess.run(loss_vector)
    4.0
    >>>loss_test = loss_vector
    >>>assert (sess.run(missing_mse(y_true_test, y_pred_test)) == sess.run(K.sum(loss_test)))
    """

    bool_missing = tf.logical_not(tf.equal(y_true,-1.0))
    float_missing = K.cast(bool_missing, dtype='float32')
    missing_squared = K.square( (y_pred - y_true)*float_missing )
    masked = tf.boolean_mask(missing_squared, bool_missing)
    loss_vector = K.mean(masked, axis=-1)
    return K.sum(loss_vector)

if __name__ == '__main__':
    import doctest
    doctest.testmod()