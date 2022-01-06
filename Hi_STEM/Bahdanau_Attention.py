import tensorflow as tf

def attention(inputs, attention_size, time_major=False, return_alphas=False):


    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    w1 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    semantic_encoding = tf.reduce_mean(v, axis=1)

    temp1 = tf.tensordot(v, w1, axes=1)
    print("temp1 shape", temp1.shape)

    scores = tf.tanh(tf.tensordot(v, w1, axes=1) + tf.tensordot(semantic_encoding, w2, axes=1))

    print("scores shape", scores.shape)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector

    alphas = tf.nn.softmax(scores, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    attention_name = "bahdanau attention"

    if not return_alphas:
        return output, attention_name
    else:
        return output, alphas, attention_name
