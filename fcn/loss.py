import tensorflow as tf
def loss(y_pred,y_true):

    # class_weights = np.array([
    #     0.1792, 
    #     1.8208])

    # class_weights = np.array([
    #     0.5493, 
    #     5.5755])


    epsilon = tf.constant(value=1e-4)
    y_pred = y_pred + epsilon

    y_pred = tf.reshape(y_pred, (-1, 2))

    y_true = tf.reshape(y_true, (-1, 2))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # cross_entropy = -tf.sum(y_true * K.log(softmax), axis=1)
    cross_entropy_mean = tf.reduce_sum(cross_entropy)

    return cross_entropy_mean
