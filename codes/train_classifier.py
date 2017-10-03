from parse_stanford import get_data
import pickle, numpy as np
from model import model
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

def _get_data(data):
    data_context_x = []
    data_lemma_x = []
    data_context_pos_deprel = [] # 255 * 5 = 1275 (208 Dependency, 47 POS)
    data_children_pos_deprel = [] # single layer 208, 47
    data_pos_deprel = [] #208, 47, 47 = 302
    data_y = []
    for doc in data:
        for sent_no in doc:
            sent_temp = ['PADDING', 'PADDING']
            for token_no in doc[sent_no].tokens:
                sent_temp += [doc[sent_no].tokens[token_no]]
                if doc[sent_no].tokens[token_no].event_tag == 'O': data_y.append([1,0,0])
                elif doc[sent_no].tokens[token_no].event_tag == 'B': data_y.append([0,1,0])
                else: data_y.append([0,0,1])
            sent_temp += ['PADDING', 'PADDING']
            for i in range(2, len(sent_temp) - 2):
                feat_temp = []
                for j in range(-2, 3, 1):
                    try: feat_temp += word_vecs[sent_temp[i + j].lemma.lower()]
                    except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                    except AttributeError: feat_temp += word_vecs[sent_temp[i + j]]
                data_context_x.append(feat_temp)

                feat_temp = []
                for j in range(-2, 3, 1):
                    try: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs[sent_temp[i + j].parent_deprel]
                    except KeyError: feat_temp += pos_vecs[sent_temp[i + j].POS] + dep_vecs['UNKNOWN']
                    except AttributeError: feat_temp += pos_vecs[sent_temp[i + j]] + dep_vecs[sent_temp[i + j]]
                data_context_pos_deprel.append(feat_temp)

                try: data_lemma_x.append(word_vecs[sent_temp[i].lemma.lower()])
                except KeyError: data_lemma_x.append(word_vecs['UNKNOWN_MY'])

                try: data_pos_deprel.append(pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs[sent_temp[i].parent_deprel])
                except KeyError:
                    if sent_temp[i].parent_POS: data_pos_deprel.append(pos_vecs[sent_temp[i].POS] +  pos_vecs[sent_temp[i].parent_POS] +  dep_vecs['UNKNOWN'])
                    else: data_pos_deprel.append(pos_vecs[sent_temp[i].POS] +  pos_vecs['ROOT'] +  dep_vecs['UNKNOWN'])

                child_feats = []
                child_feats_temp = [0.0]*208
                for child in sent_temp[i].children_deprel:
                    try: child_feats_temp[max( (v, i) for i, v in enumerate(dep_vecs[child]))[1]] += 1.0
                    except KeyError: pass
                child_feats += child_feats_temp

                child_feats_temp = [0.0]*47
                for child in sent_temp[i].children_POS:
                    child_feats_temp[max( (v, i) for i, v in enumerate(pos_vecs[child]))[1]] += 1.0
                child_feats += child_feats_temp

                data_children_pos_deprel.append(child_feats)

    return data_context_x, data_context_pos_deprel, data_lemma_x, data_pos_deprel, data_children_pos_deprel, data_y

def _get_joint(data):
    data_x = []
    data_y = []
    for doc in data:
        #doc = data[file]
        for sent_no in doc:
            sent_temp = ['PADDING', 'PADDING']
            for token_no in doc[sent_no].tokens:
                sent_temp += [doc[sent_no].tokens[token_no]]
                if doc[sent_no].tokens[token_no].event_realis == 'actual': data_y.append([1,0,0,0])
                elif doc[sent_no].tokens[token_no].event_realis == 'generic': data_y.append([0,1,0,0])
                elif doc[sent_no].tokens[token_no].event_realis == 'other': data_y.append([0,0,1,0])
                else: data_y.append([0, 0, 0, 1])
            sent_temp += ['PADDING', 'PADDING']
            for i in range(2, len(sent_temp) - 2):
                feat_temp = _get_prefix_sufix(sent_temp[i].word.lower()) #[]
                #for j in range(-2, 3, 1):
                    #try: feat_temp += word_vecs[sent_temp[i + j].lemma.lower()]
                    #except KeyError: feat_temp += word_vecs['UNKNOWN_MY']
                    #except AttributeError: feat_temp += word_vecs[sent_temp[i + j]]

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
    return data_x, data_y

def _train_test():
    tag_folder = '../data/2015/training/event_tags/'
    data_folder = '../data/2015/training/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    if not combined: train_data_context_x, train_data_context_pos_deprel, train_data_lemma_x, train_data_pos_deprel, train_data_children_pos_deprel, train_data_y = _get_data(data)
    else: train_x1, train_y1 = _get_joint(data)

    tag_folder = '../data/2015/eval/event_tags/'
    data_folder = '../data/2015/eval/stanford_parse/'
    data = get_data(tag_folder, data_folder)

    if not combined: test_data_context_x, test_data_context_pos_deprel, test_data_lemma_x, test_data_pos_deprel, test_data_children_pos_deprel, test_data_y = _get_data(data)
    else: train_x2, train_y2 = _get_joint(data)

    tag_folder = '../data/2016/event_tags/'
    data_folder = '../data/2016/stanford_parse/'
    data = get_data(tag_folder, data_folder)
    train_x3, train_y3 = _get_joint(data)

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    if not combined: model.fit([np.array(train_data_context_x+test_data_context_x), np.array(train_data_context_pos_deprel+test_data_context_pos_deprel), np.array(train_data_lemma_x+test_data_lemma_x),
               np.array(train_data_pos_deprel+test_data_pos_deprel), np.array(train_data_children_pos_deprel+test_data_children_pos_deprel),], np.array(train_data_y+test_data_y), batch_size=1500, nb_epoch=15, verbose=1, shuffle=True)
    else: model.fit(np.array(train_x1 + train_x2+train_x3), np.array(train_y1+train_y2+train_y3), batch_size=1000, nb_epoch=15, verbose=1, shuffle=True)


    model.save('realis_models/model_6.h5')
    """
    predicted = model.predict([np.array(test_data_context_x), np.array(test_data_lemma_x)])

    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for index,elem in enumerate(predicted):
        ind = max( (v, i) for i, v in enumerate(elem) )[1]
        if ind == 0 and test_data_y[index][0] == 1:
            TN += 1
        elif ind > 0 and test_data_y[index][0] == 1:
            FP += 1
        elif ind == 0 and (test_data_y[index][1] == 1 or test_data_y[index][2] == 1):
            FN += 1
        elif ind > 0 and (test_data_y[index][1] == 1 or test_data_y[index][2] == 1):
            TP += 1

    print TP, FP, TN, FN, len(test_data_y)
    """


_train_test()
