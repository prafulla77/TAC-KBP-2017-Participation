from keras.layers import Dense, merge, Input
from keras.models import Model
from keras import backend as K

def cos_dist(vests):
    x, y = vests[0], vests[1]
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def euc_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = x[0] - x[1]
    output = (K.sum((s ** 2), axis=1))
    output = K.reshape(output, (K.shape(output)[0],1))
    return output

def abs_dist(x):
    'Merge function: euclidean_distance(u,v)'
    s = abs(x[0] - x[1])
    output = (K.sum((s), axis=1))
    output = K.reshape(output, (K.shape(output)[0],1))
    return output

def cos_dist_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def euc_dist_shape(input_shape):
    'Merge output shape'
    shape = list(input_shape)
    outshape = (shape[0][0],1)
    return tuple(outshape)

Dense_lemma_layer = Dense(347, activation='sigmoid')
Dense_lemma_in_1 = Input(shape=(347,))
Dense_lemma_in_2 = Input(shape=(347,))
Dense_lemma_out_1 = Dense_lemma_layer(Dense_lemma_in_1)
Dense_lemma_out_2 = Dense_lemma_layer(Dense_lemma_in_2)

Dense_other_layer = Dense(380, activation='sigmoid')
Dense_other_in_1 = Input(shape=(380,))
Dense_other_out_1 = Dense_other_layer(Dense_other_in_1)
Dense_other_out_1 = Dense(50, activation='sigmoid')(Dense_other_out_1)

merged_layer_1 = merge([Dense_lemma_out_1, Dense_lemma_out_2], mode=euc_dist, output_shape=euc_dist_shape)
merged_layer_2 = merge([Dense_lemma_out_1, Dense_lemma_out_2], mode=cos_dist, output_shape=cos_dist_shape)
merged_layer_3 = merge([Dense_lemma_out_1, Dense_lemma_out_2], mode=abs_dist, output_shape=euc_dist_shape)

merged_layer = merge([merged_layer_1, merged_layer_2, merged_layer_3, Dense_other_out_1], mode='concat', concat_axis=-1)

predictions_1 = Dense(10, activation='sigmoid')(merged_layer)
predictions = Dense(1, activation='sigmoid')(predictions_1)
model = Model(input=[Dense_lemma_in_1, Dense_lemma_in_2, Dense_other_in_1], output=predictions)

print model.summary()
