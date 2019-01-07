'''
In this paper, the authors choose CNN as their discriminator as CNN has recently been shown of great effectiveness in text (token sequence) classification.
'''
import tensorflow as tf
import numpy as np

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear_function(inp, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    '''
    shape = inp.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError(f"Linear is expecting 2D arguments: {str(shape)}")
    if not shape[1]:
        raise ValueError(f"Linear expects shape[1] of arguments: {str(shape)}")

    input_size = shape[1]

    # Computation
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=inp.dtype)
        bias = tf.get_variable("Bias", [output_size], dtype=inp.dtype)

    return tf.matmul(inp, tf.transpose(matrix)) + bias

def highway(inp, size, num_layers=1, bias=2.0, f=tf.nn.relu, scope="Highway"):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope(scope):
        for i in range(num_layers):
            g = f(linear_function(inp, size, scope=f'highway_lin_{i}'))
            t = tf.sigmoid(linear_function(inp, size, scope=f'highway_lin_{i}') + bias)

            output = t*g + (1. - t)*inp
            inp = output

    return output

class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # seq_len – The length of our sentences. It is 20 in this paper
    # num_classes – Number of classes in the output layer
    # vocab_size – The size of our vocabulary. This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size].
    # embedding_size – The dimensionality of our embeddings.
    # filter_sizes – The number of words we want our convolutional filters to cover
    # num_filters – The number of filters per filter size.
    def __init__(self, seq_len, num_classes, vocab_size, emb_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        # The first dimension is the batch size, and using None allows the network to handle arbitrarily sized batches.
        self.input_x = tf.placeholder(tf.int32, [None, seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer - tf.device("/cpu:0") forces an operation to be executed on the CPU. By default TensorFlow will try to put the operation on the GPU if one is available, but the embedding implementation doesn’t currently have GPU support and throws an error if placed on the GPU.
            # tf.name_scope creates a new Name Scope with the name “embedding”. The scope adds all operations into a top-level node called “embedding” so that we get a nice hierarchy when visualizing our network in TensorBoard.
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                # W is embedding matrix that we learn during training. We initialize it using a random uniform distribution.
                # tf.nn.embedding_lookup creates the actual embedding operation. The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].
                # TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to batch, width, height and channel. The result of the embedding doesn’t contain the channel dimension, so we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
                self.W = tf.Variable(tf.random_uniform([vocab_size, emb_size], -1.0, 1.0), name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Convolution + Maxpooling for each filter size.
            # Because each convolution produces tensors of different shapes, we need to iterate through them, create a layer for each of them and then merge the results into a big feature vector
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope(f"conv-maxpool-{filter_size}"):
                    # Convolution layer
                    # W - filter matrix, h - result of applying nonlinearity to convolution output. Each filter slides over the whole embedding, but varies in how many words it covers. "VALID" padding means we slide over our sentence without padding the edges, performing a narrow convolution that gives an output of shape [1, seq_len - filter_size + 1, 1, 1].
                    filter_shape = [filter_size, emb_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                        W,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv")
                    # Apply non-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Max-pooling
                    # Performing max-pooling over the output of a specific filter size leaves us with a tensor of shape [batch_size, 1, 1, num_filters]. This is essentially a feature vector, where the last dimension corresponds to our features.
                    pooled = tf.nn.max_pool(h,
                                            ksize=[1, seq_len - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding="VALID",
                                            name="pool")
                    pooled_outputs.append(pooled)

            # Combine all pooled features
            # Once we have all the pooled output tensors from each filter size we combine them into one long feature vector of shape [batch_size, num_filters_total]. Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible.
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                # Using the feature vector from max-pooling (with dropout applied) we can generate predictions by doing a matrix multiplication and picking the class with the highest score.
                # tf.nn.xw_plus_b is a convenience wrapper to perform the Wx + b matrix multiplication.
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits is a convenience function that calculates the cross-entropy loss for each class, given our scores and the correct input labels. We then take the mean of the losses.
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda + l2_loss

            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            d_optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
            self.train_op = d_optimizer.apply_gradients(grads_and_vars)
