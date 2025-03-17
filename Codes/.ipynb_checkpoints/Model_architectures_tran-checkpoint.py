import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, Embedding, Input
from tensorflow.keras.models import Model

## training= True/False whether we have drop out or not

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def transformer_model(num_classes, training, model_cbow, num_transformer_blocks=4, head_size=128, num_heads=4, ff_dim=512, dropout=0.5):
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    
    embedding_layer = Embedding(len(weight_vectors),
                                weight_vectors.shape[1],
                                weights=[weight_vectors],
                                mask_zero=False,
                                trainable=False)

    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode='int',
        vocabulary=model_cbow.wv.index_to_key,
        standardize=None
    )
    
    input_layer = Input(shape=(1,), dtype=tf.string)
    vectorize = vectorize_layer(input_layer)
    emb_output = embedding_layer(vectorize)

    x = emb_output
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    pred_output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=pred_output)
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  metrics=["accuracy"])
    
    return model



## model_cbow: word embedding model, use to transfer gene name to embedding vector
def simple_lstm(num_classes, training, model_cbow):
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)
    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape

    lstm_layer = tf.keras.layers.LSTM(100,  dropout = 0.6)

    lstm_output = lstm_layer(emb_output, training = True)

    dropout_layer = tf.keras.layers.Dropout(0.6)(lstm_output, training = True)

    pred_head = tf.keras.layers.Dense(num_classes)

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                 metrics= "accuracy")
    
    return model



def attention_lstm_model(num_classes, training, model_cbow): 
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)

    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape

    lstm_layer = tf.keras.layers.LSTM(100, return_sequences = True, dropout = 0.5)

    lstm_output = lstm_layer(emb_output, training = True)

    x_a = tf.keras.layers.Dense(lstm_output.get_shape()[-1]//2, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(lstm_output) 
    
    x_a = tf.keras.layers.Dropout(0.5)(x_a, training = True)
    
    x_a = tf.keras.layers.Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(x_a)

    x_a = tf.keras.layers.Flatten()(x_a)

    att_out = tf.keras.layers.Activation('softmax')(x_a) 

    x_a2 = tf.keras.layers.RepeatVector(lstm_output.get_shape()[-1])(att_out)

    x_a2 = tf.keras.layers.Permute([2,1])(x_a2)

    out = tf.keras.layers.Multiply()([lstm_output,x_a2])
    
    out = tf.keras.layers.Lambda(lambda x : tf.math.reduce_sum(x, axis = 1), name='expectation_over_words')(out)
    
    dropout_layer = tf.keras.layers.Dropout(0.65)(out, training = True)

    pred_head = tf.keras.layers.Dense(num_classes)

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                 metrics= "accuracy")
    
    return model


def non_recurrent_attention_model(num_classes, training, model_cbow): 
    padding_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    unknown_vector = np.zeros((1, model_cbow.wv.vectors.shape[1]))
    weight_vectors = np.vstack((padding_vector, unknown_vector))
    weight_vectors = np.vstack((weight_vectors, model_cbow.wv.vectors))
    embedding_layer = tf.keras.layers.Embedding(len(weight_vectors),
                            weight_vectors.shape[1],
                            weights=[weight_vectors],
                            mask_zero = False,
                            trainable=False)
    
    vectorize_layer = tf.keras.layers.TextVectorization(
                     output_mode='int',
                     vocabulary=model_cbow.wv.index_to_key, 
                     standardize = None)
    input_layer = tf.keras.layers.Input(shape = (1,), dtype = tf.string)

    vectorize = vectorize_layer(input_layer)

    vectorize.shape

    emb_output = embedding_layer(vectorize)

    emb_output.shape

    x_a = tf.keras.layers.Dense(emb_output.get_shape()[-1]//2, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(emb_output) 
    
    x_a = tf.keras.layers.Dropout(0.5)(x_a, training = True)
    
    x_a = tf.keras.layers.Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(x_a)

    x_a = tf.keras.layers.Flatten()(x_a)

    att_out = tf.keras.layers.Activation('softmax')(x_a) 

    x_a2 = tf.keras.layers.RepeatVector(emb_output.get_shape()[-1])(att_out)

    x_a2 = tf.keras.layers.Permute([2,1])(x_a2)

    out = tf.keras.layers.Multiply()([emb_output,x_a2])
    
    out = tf.keras.layers.Lambda(lambda x : tf.math.reduce_sum(x, axis = 1), name='expectation_over_words')(out)

    dropout_layer = tf.keras.layers.Dropout(0.65)(out, training = True)

    pred_head = tf.keras.layers.Dense(num_classes)

    pred_output = pred_head(dropout_layer)

    model = tf.keras.models.Model(input_layer, pred_output)
    
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                 optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                 metrics= "accuracy")
    
    return model