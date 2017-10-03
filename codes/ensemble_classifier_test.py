from parse_stanford_test import get_data
import pickle, numpy as np
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
tf.python.control_flow_ops = tf

np.random.seed(0)

NER_map = {'PERSON':0, 'LOCATION':1, 'ORGANIZATION':2, 'MISC':3,'MONEY':4, 'NUMBER':5, 'ORDINAL':6, 'PERCENT':7}

TYPE_map = {0:'Conflict_Attack',1:'Conflict_Demonstrate',2:'Contact_Meet',3:'Contact_Correspondence',
            4:'Contact_Broadcast',5:'Contact_Contact',6:'Life_Injure',7:'Life_Die',8:'Justice_Arrest-Jail',
            9:'Manufacture_Artifact',11:'Movement_Transport-Artifact',10:'Movement_Transport-Person',
            12:'Personnel_Elect',13:'Personnel_End-Position',14:'Personnel_Start-Position',15:'Transaction_Transfer-Money',
            16:'Transaction_Transfer-Ownership',17:'Transaction_Transaction'}

with open('../vocab/2017_.pkl', 'rb') as fp:
    word_vecs = pickle.load(fp)
with open('../vocab/POS.pkl', 'rb') as fp:
    pos_vecs = pickle.load(fp)
with open('../vocab/deprel.pkl', 'rb') as fp:
    dep_vecs = pickle.load(fp)

word_vecs['PADDING'] = [0.0]*300
word_vecs['UNKNOWN_MY'] = [0.50]*300

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
    for doc in data:
        for sent_no in doc:
            for token_no in doc[sent_no].tokens:

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
    print len(data_x)
    return data_x
def _get_joint_test(data):
    data_x = []
    filenames = []
    for doc in data:
        for sent_no in doc:
            sent_temp = ['PADDING', 'PADDING']
            for token_no in doc[sent_no].tokens:
                filenames.append(doc[sent_no].tokens[token_no])
                sent_temp += [doc[sent_no].tokens[token_no]]
            sent_temp += ['PADDING', 'PADDING']
            for i in range(2, len(sent_temp) - 2):
                feat_temp = _get_prefix_sufix(sent_temp[i].word.lower()) #[]

                for j in range(-2, 3, 1):
                    try: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs[sent_temp[i + j].parent_deprel]
                    except KeyError: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs['UNKNOWN']
                    except AttributeError: feat_temp += pos_vecs[sent_temp[i + j]] + dep_vecs[sent_temp[i + j]]

                if 1:
                    try: feat_temp += list(np.array(word_vecs[sent_temp[i].word.lower()]) - np.array(word_vecs[sent_temp[i].lemma.lower()]))
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                if 1:
                    try: feat_temp += word_vecs[sent_temp[i].lemma.lower()]
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']

                try: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs[sent_temp[i].parent_deprel]
                except KeyError:
                    if sent_temp[i].parent_POS: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs['UNKNOWN']
                    else: feat_temp += pos_vecs[sent_temp[i].POS] +  pos_vecs['ROOT'] +  dep_vecs['UNKNOWN']

                child_feats_temp = [0.0]*208
                for child in sent_temp[i].children_deprel:
                    try: child_feats_temp[max( (v, i) for i, v in enumerate(dep_vecs[child]))[1]] += 1.0
                    except KeyError: pass
                feat_temp += child_feats_temp

                child_feats_temp = [0.0]*47
                for child in sent_temp[i].children_POS:
                    child_feats_temp[max( (v, i) for i, v in enumerate(pos_vecs[child]))[1]] += 1.0
                feat_temp += child_feats_temp

                data_x.append(feat_temp)
    return data_x, filenames

from keras.models import load_model
import os
from collections import OrderedDict as OD

coref_model = my_model = load_model('coref_model.h5')

def _get_arg_feat(e1,e2):
    temp = [0.0]*8
    for i in range(8):
        if e1.args[i] == e2.args[i] and len(e1.args[i]) > 0: temp[i] = 1.0
        elif e1.args[i] != e2.args[i] and len(e1.args[i]) > 0: temp[i] = -1.0
    return temp

def _get_coref_feat(e1, e2):
    data_x_1 = []
    data_x_2 = []
    data_x_3 = []
    if 0:
        try: temp = pos_vecs[e1.POS] + word_vecs[e1.lemma.lower()]
        except: temp = pos_vecs[e1.POS] + word_vecs['UNKNOWN_MY']
        try: temp += word_vecs[e1.word.lower()]
        except: temp += word_vecs['UNKNOWN_MY']  #300
        data_x_1.append(temp)

        try: temp = pos_vecs[e2.POS] + word_vecs[e2.lemma.lower()]
        except: temp = pos_vecs[e2.POS] + word_vecs['UNKNOWN_MY']
        try: temp += word_vecs[e2.word.lower()]
        except: temp += word_vecs['UNKNOWN_MY']  #300
        data_x_2.append(temp)
    if 1:
        try: data_x_1.append(pos_vecs[e1.POS] + word_vecs[e1.lemma.lower()])
        except: data_x_1.append(pos_vecs[e1.POS] + word_vecs['UNKNOWN_MY'])   #347
        try: data_x_2.append(pos_vecs[e2.POS] + word_vecs[e2.lemma.lower()])
        except: data_x_2.append(pos_vecs[e2.POS] + word_vecs['UNKNOWN_MY'])

    data_x_3.append(_get_arg_feat(e1, e2)  + _get_prefix_sufix(e1.word.lower()) + _get_prefix_sufix(e2.word.lower()))    # 80
    return [np.array(data_x_1), np.array(data_x_2), np.array(data_x_3)]

def _get_coref(all_events):
    cls = {}
    cls_no = 1
    inv_cls = {}
    keys = all_events.keys()
    for ev_ind, ev in enumerate(keys):
        best_match = ''
        max_score = 0.0
        for prev_id in range(ev_ind):
            score = coref_model.predict(_get_coref_feat(all_events[keys[prev_id]], all_events[ev])) #ev[1] is event id
            if score > max_score:
                best_match = keys[prev_id]
                max_score = score
        if max_score > 0.5:
            cls[inv_cls[best_match]].append(ev)
            inv_cls[ev] = inv_cls[best_match]
        else:
            cls_no += 1
            cls[cls_no] = [ev]
            inv_cls[ev] = cls_no
    ans = []
    for elem in cls.values():
        if len(elem) > 1:
            ans.append(elem)
    return ans

def _train_test():
    data_folder = '../data/2017/out/'
    data = get_data(data_folder)

    models = os.listdir('type_models/')
    type_prediction = []
    type_test_x = _get_joint(data)
    for model in models:
        if ".DS" not in model:
            print model
            my_model = load_model('type_models/'+model)
            type_prediction.append(my_model.predict(np.array(type_test_x)))
    type_predict = type_prediction[0]
    for pred in type_prediction[1:]:
        type_predict += pred

    models = os.listdir('realis_models/')
    prediction = []
    test_x, filenames = _get_joint_test(data)

    for model in models:
        if ".DS" not in model:
            print model
            my_model = load_model('realis_models/'+model)
            prediction.append(my_model.predict(np.array(test_x)))

    predicted = prediction[0]
    for pred in prediction[1:]:
        predicted += pred

    file = ''
    predicted = list(predicted)
    print len(predicted), len(filenames)
    f = open('System_III.tbf', 'w')
    ev_id = 0
    all_events = OD()
    for i in range(len(filenames)):
        if type_predict[i][18] > 9.5: continue #2.95
        else: type_predict[i][18] = 0.0
        if filenames[i].filename[:-4] != file:
            file = filenames[i].filename[:-4]
            if ev_id!= 0:
                # coerference resolution
                corefs = _get_coref(all_events)
                cid = 1111
                for cls in corefs:
                    cid += 1
                    f.write("@Coreference\tC"+str(cid)+'\t')
                    for eid in cls[:-1]:
                        f.write(eid+',')
                    f.write(cls[-1]+'\n')
                f.write('#EndOfDocument\n')
                all_events = OD()
            f.write('#BeginOfDocument '+file+'\n')
        if predicted[i][-1] < 3.75: #6.1
            predicted[i][-1] = 0
        ind = max((v, i) for i, v in enumerate(predicted[i]))[1]
        type_ind = max((v, i) for i, v in enumerate(type_predict[i]))[1]

        if ind == 0:
            ev_id += 1
            all_events['E'+str(ev_id)] = filenames[i]
            f.write('System1\t'+file+'\tE'+str(ev_id)+'\t'+str(filenames[i].CharacterOffsetBegin)+','+str(filenames[i].CharacterOffsetEnd)+'\t'+filenames[i].word
                    +'\t'+TYPE_map[type_ind]+'\t'+'Actual\n')
        if ind == 1:
            ev_id += 1
            all_events['E'+str(ev_id)] = filenames[i]
            f.write('System1\t'+file+'\tE'+str(ev_id)+'\t'+str(filenames[i].CharacterOffsetBegin)+','+str(filenames[i].CharacterOffsetEnd)+'\t'+filenames[i].word
                    +'\t'+TYPE_map[type_ind]+'\t'+'Generic\n')
        if ind == 2:
            ev_id += 1
            all_events['E'+str(ev_id)] = filenames[i]
            f.write('System1\t'+file+'\tE'+str(ev_id)+'\t'+str(filenames[i].CharacterOffsetBegin)+','+str(filenames[i].CharacterOffsetEnd)+'\t'+filenames[i].word
                    +'\t'+TYPE_map[type_ind]+'\t'+'Other\n')
    corefs = _get_coref(all_events)
    cid = 1111
    for cls in corefs:
        cid += 1
        f.write("@Coreference\tC"+str(cid)+'\t')
        for eid in cls[:-1]:
            f.write(eid+',')
        f.write(cls[-1]+'\n')
    f.write('#EndOfDocument\n')
    f.close()


_train_test()
