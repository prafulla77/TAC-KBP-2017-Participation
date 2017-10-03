from keras.models import Model
from keras.layers.core import  Dropout
from keras.layers import Dense, Input, merge
import tensorflow as tf
from keras import regularizers

tf.python.control_flow_ops = tf

if 0:
    context_ip = Input(shape=(1500,))
    context_dense = Dense(50, activation='sigmoid')(Dropout(0.5)(Dense(1500, activation='relu')(context_ip)))

    context_pos_dep_ip = Input(shape=(1275,))
    context_dense_pos_dep = Dense(50, activation='sigmoid')(Dropout(0.25)(Dense(1275, activation='relu')(context_pos_dep_ip)))

    lemma_ip = Input(shape=(300,))
    lemma_dense = Dense(50, activation='sigmoid')(Dropout(0.05)(Dense(300, activation='relu')(lemma_ip)))

    pos_deprel_ip = Input(shape=(302,))
    pd_dense = Dense(50, activation='sigmoid')(Dropout(0.05)(Dense(302, activation='relu')(pos_deprel_ip)))

    child_pos_deprel_ip = Input(shape=(255,))
    cpd_dense = Dense(50, activation='sigmoid')(Dropout(0.05)(Dense(255, activation='relu')(child_pos_deprel_ip)))

    merged_layer = merge([context_dense, context_dense_pos_dep, lemma_dense, pd_dense, cpd_dense], mode='concat', concat_axis=-1)

    predictions = Dense(3, activation='softmax')(Dense(50, activation='sigmoid')(Dense(250, activation='relu')(merged_layer)))

    model = Model(input=[context_ip, context_pos_dep_ip, lemma_ip, pos_deprel_ip, child_pos_deprel_ip], output=predictions)
    print model.summary()

else:
    context_ip = Input(shape=(2468,))
    context_dense = Dense(1234, activation='relu')(context_ip)
    #context_dense = Dropout(0.1)(context_dense)
    context_dense = Dropout(0.5)(context_dense)
    context_dense = Dense(600, activation='sigmoid')(context_dense)
    #context_dense = Dropout(0.4)(context_dense)
    context_dense = Dense(200, activation='sigmoid')(context_dense) #300
    #context_dense = Dropout(0.4)(context_dense)
    context_dense = Dense(50, activation='sigmoid')(context_dense) #30
    predictions = Dense(4, activation='softmax')(context_dense) #4

    model = Model(input=context_ip, output=predictions)
    print model.summary()
