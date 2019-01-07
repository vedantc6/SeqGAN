import keras.backend as K
import numpy as np
from keras.layers import Input, Dense, Conv2d, Embedding, MaxPool2d, Add, Multiply, Lambda, Dropout, Activation
from keras.layers.merge import concatenate
from keras import initializers
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

def linear(input_, output_size):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))

    input_size = shape[1]

    matrix = K.placeholder([output_size, input_size], dtype=input_.dtype)
    bias_term = K.placeholder([output_size], dtype=input_.dtype)
    return K.dot(input_, K.transpose(matrix)) + bias_term

def highway(input_, num_layers=1, bias=2.0, activation='tanh'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity (or activation), t is transform gate, and (1 - t) is carry gate.
    """
    dim = K.int_shape(input_)[-1]
    gate_bias_initializer = initializers.Constant(bias)
    for i in range(num_layers):
        # Compute t = sigmoid(Wy + b)
        t = Dense(units=dim,
                  bias_initializer=gate_bias_initializer,
                  activation='sigmoid')(input_)

        # Compute g(Wy + b) and then t*g(Wy + b), g is an activation
        g = Dense(units=dim, activation=activation)(input_)
        transformed_gate = Multiply()([t, g])

        # Compute (1 - t)*y
        neg_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim, ))(t)
        carry_gate = Multiply()([neg_gate, input_])
        input_ = Add()([transformed_gate, carry_gate])

    return input_

class Discriminator(object):
    '''
    A CNN for text classification
    Uses an embedding layer, followed by convolutional, max_pooling and softmax layer
    Args:
      max_sequence_length: the length of a sequence
      num_classes: the dimensionality of output vector, i.e., number of classes
      vocab_size: the number of vocabularies
      emb_size: the dimensionality of an embedding vector
      filter_sizes: filter size
      num_filters: number of filter
      l2_lambda: lambda, a parameter for L2 regularizer
    '''
    def __init__(self, max_sequence_length, num_classes, vocab_size, emb_size, filter_sizes, num_filters, l2_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = Input(shape=(max_sequence_length, ), dtype='int32')
        self.input_y = Input(shape=(num_classes, ), dtype='int32')
        self.dropout = Input(shape=1, dtype='float32')

        # Embedding layer
        self.embed_chars = Embedding(input_dim=vocab_size,
                                    output_dim=emb_size,
                                    input_length=max_sequence_length,
                                    embedding_initializer='uniform')(self.input_x)
        self.embed_chars_expanded = K.expand_dims(self.embed_chars, -1)


        # Convolution + Max Pooling layer for each filter size
        pooled_outputs = []
        for fsize, nfilter in zip(filter_sizes, num_filters):
            # Convolution
            conv = Conv2d(filters=nfilter, kernel_size=fsize, padding='valid', activation='relu', strides=1)(self.embed_chars_expanded)
            # Max Pooling
            conv = MaxPool2d(pool_size=2)(conv)
            pooled_outputs.append(conv)

        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        self.h_pool = concatenate(pooled_outputs, 3)
        self.h_pool_flat = K.reshape(self.h_pool, [-1, num_filters_total])

        # Add highway
        self.h_highway = highway(self.h_pool_flat, num_layers=1, bias=0, activation='relu')

        # Add dropout
        self.h_drop = Dropout(self.dropout)(self.h_highway)

        # Final (unnormalized scores) and predictions
        self.scores = BatchNormalization()(self.h_drop)
        self.preds = Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(self.h_drop)

        # Train the model
        adam = Adam(lr=1e-4)
        model = Model(inputs=self.input_x, outputs=self.preds)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
        model.summary()
