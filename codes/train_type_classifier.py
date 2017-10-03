from parse_stanford import get_data
import pickle, numpy as np
from sklearn.metrics import recall_score, precision_score
#np.random.seed(0)
from keras.models import Model
from keras.layers.core import  Dropout
from keras.layers import Dense, Input, merge
import tensorflow as tf

tf.python.control_flow_ops = tf

with open('../vocab/2015_training_.pkl', 'rb') as fp:
    word_vecs = pickle.load(fp)
with open('../vocab/2015_eval_.pkl', 'rb') as fp:
    word_vecs.update(pickle.load(fp))
with open('../vocab/2016_.pkl', 'rb') as fp:
    word_vecs.update(pickle.load(fp))
with open('../vocab/deprel.pkl', 'rb') as fp:
    dep_vecs = pickle.load(fp)

word_vecs['PADDING'] = [0.0]*300
word_vecs['UNKNOWN_MY'] = [0.50]*300

NER_map = {'PERSON':0, 'LOCATION':1, 'ORGANIZATION':2, 'MISC':3,'MONEY':4, 'NUMBER':5, 'ORDINAL':6, 'PERCENT':7}

def _get_prefix_sufix(word):
    suffix = ['te', 'tor', 'or', 'ing', 'cy', 'id', 'ed', 'en', 'er', 'ee', 'pt', 'de', 'on', 'ion', 'tion', 'ation',
              'ction', 'de', 've', 'ive', 'ce', 'se', 'ty', 'al', 'ar', 'ge', 'nd', 'ize', 'ze', 'it', 'lt'] #31
    prefix = ['re', 'in', 'at', 'tr', 'op'] #5
    ans = []
    for suf in suffix:
        if word[-len(suf):] == suf: ans.append(1.0)
        else: ans.append(0.0)
    for pref in prefix:
        if word[:len(pref)] == pref: ans.append(1.0)
        else: ans.append(0.0)
    return ans

def _get_joint(data):
    data_x = []
    data_y = []
    for doc in data:
        for sent_no in doc:
            for token_no in doc[sent_no].tokens:
                temp = [0]*19
                if doc[sent_no].tokens[token_no].event_subtype == None: continue
                else:
                    for type_index, type in enumerate(['attack', 'demonstrate', 'meet', 'correspondence', 'broadcast', 'contact',
                        'injure', 'die', 'arrestjail', 'artifact', 'transportperson', 'transportartifact', 'elect', 'endposition', 'startposition',
                        'transfermoney', 'transferownership', 'transaction']):
                        if doc[sent_no].tokens[token_no].event_subtype == type:
                            temp[type_index] = 1
                if sum(temp) != 1: temp[18] = 1#print temp, doc[sent_no].tokens[token_no].event_subtype
                data_y.append(temp)

                feat_temp = _get_prefix_sufix(doc[sent_no].tokens[token_no].word.lower())

                if 1:
                    try: feat_temp += word_vecs[doc[sent_no].tokens[token_no].word.lower()]
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                if 1:
                    try: feat_temp += word_vecs[doc[sent_no].tokens[token_no].lemma.lower()]
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']

                child_feats_temp = [0.0]*208
                for child in doc[sent_no].tokens[token_no].children_deprel:
                    try: child_feats_temp[max( (v, i) for i, v in enumerate(dep_vecs[child]))[1]] += 1.0
                    except KeyError: pass
                feat_temp += child_feats_temp

                child_feats_temp = [0.0] * 8 #PERSON, LOCATION, ORGANIZATION, MISC, MONEY, NUMBER, ORDINAL, PERCENT
                for child in doc[sent_no].tokens[token_no].children_token_nos:
                    if doc[sent_no].tokens[child].NER in NER_map:
                        child_feats_temp[NER_map[doc[sent_no].tokens[child].NER]] += 1.0
                feat_temp += child_feats_temp

                # NER related features
                data_x.append(feat_temp)
    print len(data_x), len(data_y)
    return data_x, data_y

def _train_test():
    tag_folder = '../data/2015/training/event_tags/'
    data_folder = '../data/2015/training/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    train_x1, train_y1 = _get_joint(data)

    tag_folder = '../data/2015/eval/event_tags/'
    data_folder = '../data/2015/eval/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    train_x2, train_y2 = _get_joint(data)

    # 816 dimensional features
    context_ip = Input(shape=(852,))
    context_dense = Dense(2500, activation='relu')(context_ip)
    context_dense = Dense(1400, activation='sigmoid')(context_dense)
    context_dense = Dense(700, activation='sigmoid')(context_dense)
    context_dense = Dense(200, activation='sigmoid')(context_dense)
    predictions = Dense(19, activation='softmax')(context_dense)
    model = Model(input=context_ip, output=predictions)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    tag_folder = '../data/2016/event_tags/'
    data_folder = '../data/2016/stanford_parse/'
    data = get_data(tag_folder, data_folder)
    test_x, test_y = _get_joint(data)

    model.fit(np.array(train_x1 + train_x2 + test_x), np.array(train_y1+train_y2+test_y), batch_size=500, nb_epoch=20, verbose=1, shuffle=True)

    model.save('type_models/model_type_10.h5')


_train_test()
