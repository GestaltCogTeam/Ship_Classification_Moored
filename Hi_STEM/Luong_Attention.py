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

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    semantic_encoding = tf.reduce_mean(v, axis=1)
    print("semantic_encoding shape", semantic_encoding.shape)

    w = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))
    vw = tf.tensordot(v, w, axes=1, name="vw")

    print(vw.shape)

    scores = tf.tensordot(vw, tf.transpose(semantic_encoding, [1, 0]), axes=1, name="scores")

    print(scores.shape)
    alphas = tf.nn.softmax(scores, name="alphas")

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * alphas, 1)

    attention_name = "luong attention"

    if not return_alphas:
        return output, attention_name
    else:
        return output, alphas, attention_name
