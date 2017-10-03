from parse_stanford import get_data
import pickle, numpy as np
from coref_model import model
from sklearn.metrics import precision_score, recall_score
np.random.seed(0)

combined = 1
with open('../vocab/2015_training_.pkl', 'rb') as fp:
    word_vecs = pickle.load(fp)
with open('../vocab/2015_eval_.pkl', 'rb') as fp:
    word_vecs.update(pickle.load(fp))
with open('../vocab/2016_.pkl', 'rb') as fp:
    word_vecs.update(pickle.load(fp))
with open('../vocab/POS.pkl', 'rb') as fp:
    pos_vecs = pickle.load(fp)
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

def _get_arg_feat(e1,e2):
    temp = [0.0]*8
    for i in range(8):
        if e1.args[i] == e2.args[i] and len(e1.args[i]) > 0: temp[i] = 1.0
        elif e1.args[i] != e2.args[i] and len(e1.args[i]) > 0: temp[i] = -1.0
    return temp

def _get_joint(data):
    data_x_1 = []
    data_x_2 = []
    data_x_3 = []
    data_y = []
    for doc in data:
        all_events = []
        for sent_no in doc:
            for token_no in doc[sent_no].tokens:
                if doc[sent_no].tokens[token_no].event_subtype in ['attack', 'demonstrate', 'meet', 'correspondence', 'broadcast', 'contact',
                        'injure', 'die', 'arrestjail', 'artifact', 'transportperson', 'transportartifact', 'elect', 'endposition', 'startposition',
                        'transfermoney', 'transferownership', 'transaction']:
                    all_events.append(doc[sent_no].tokens[token_no])

        for prev_ind, cur_event in enumerate(all_events):
            for ind in range(prev_ind):
                try: data_x_1.append(pos_vecs[all_events[ind].POS] + word_vecs[all_events[ind].lemma.lower()])
                except: data_x_1.append(pos_vecs[all_events[ind].POS] + word_vecs['UNKNOWN_MY'])   #347
                try: data_x_2.append(pos_vecs[cur_event.POS] + word_vecs[cur_event.lemma.lower()])
                except: data_x_2.append(pos_vecs[cur_event.POS] + word_vecs['UNKNOWN_MY'])
                try: temp = list(np.array(word_vecs[all_events[ind].lemma.lower()])-np.array(word_vecs[cur_event.lemma.lower()]))
                except KeyError: temp = [0.0]*300
                data_x_3.append(temp + _get_arg_feat(all_events[ind], cur_event) + _get_prefix_sufix(all_events[ind].word.lower()) +
                                _get_prefix_sufix(cur_event.word.lower()))    # 80
                data_y.append(int(all_events[ind].coref_id == cur_event.coref_id))
    print len(data_x_1), len(data_x_3), len(data_x_2), len(data_y)
    return data_x_1, data_x_2, data_x_3, data_y

def _train_test():
    tag_folder = '../data/2015/training/event_tags/'
    data_folder = '../data/2015/training/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    train_x1_1, train_x1_2, train_x1_3, train_y1 = _get_joint(data)

    tag_folder = '../data/2015/eval/event_tags/'
    data_folder = '../data/2015/eval/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    train_x2_1, train_x2_2, train_x2_3, train_y2 = _get_joint(data)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.fit([np.array(train_x1_1+train_x2_1), np.array(train_x1_2+train_x2_2), np.array(train_x1_3+train_x2_3)],
              np.array(train_y1+train_y2), batch_size=500, nb_epoch=35, verbose=1, shuffle=True)

    model.save('coref_model_1.h5')

_train_test()
