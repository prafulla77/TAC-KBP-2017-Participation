import xml.etree.ElementTree as ET
from collections import OrderedDict as OD
import pickle, os

NER_map = {'PERSON':0, 'LOCATION':1, 'ORGANIZATION':2, 'MISC':3,'MONEY':4, 'NUMBER':5, 'ORDINAL':6, 'PERCENT':7}

class Sentence(object):
    def __init__(self, sent_no):
        self.sent_no = sent_no
        self.tokens = OD()
        self.basic_dependencies = {}
        self.collapsed_dependencies = {}
        self.collapsed_ccprocessed_dependencies = {}

    def _add_token(self, token_id, word_obj):
        self.tokens[token_id] = word_obj

    def _add_dependency(self, type, dependency_relation):
        if type == 'basic-dependencies': self.basic_dependencies = dependency_relation
        elif type == 'collapsed-dependencies': self.collapsed_dependencies = dependency_relation
        elif type == 'collapsed-ccprocessed-dependencies': self.collapsed_ccprocessed_dependencies = dependency_relation

class Word(object):
    def __init__(self, token_no, filename):
        self.token_no = token_no
        self.word = None
        self.lemma = None
        self.CharacterOffsetBegin = None
        self.CharacterOffsetEnd = None
        self.POS = None
        self.NER = None
        self.Speaker = None
        self.event_tag = 'O'
        self.event_id = None
        self.event_type = None
        self.event_subtype = None
        self.event_realis = None
        self.parent_deprel = None
        self.parent_POS = None
        self.children_deprel = []
        self.children_POS = []
        self.children_token_nos = []
        self.entity_coref_id = None
        self.filename = filename
        self.coref_id = None
        self.args = {key: [] for key in range(8)}

    def _add_feats(self, type, value):
        if type == 'word': self.word = value
        elif type == 'lemma': self.lemma = value
        elif type == 'CharacterOffsetBegin': self.CharacterOffsetBegin = int(value)
        elif type == 'CharacterOffsetEnd': self.CharacterOffsetEnd = int(value)
        elif type == 'POS': self.POS = value
        elif type == 'NER': self.NER = value
        elif type == 'Speaker': self.Speaker = value

    def print_tag(self):
        print self.event_tag, self.event_id, self.event_type, self.event_subtype, self.event_realis

    def __str__(self):
        return (self.CharacterOffsetBegin + '\t' + self.CharacterOffsetEnd + '\t' + self.NER)

def get_data(tag_folder, data_folder):
    filenames = os.listdir(tag_folder)
    training_data = []
    for filename in filenames:
        #if filename != '2c7afa4a988d807becade727a769da50.txt': continue
        file = data_folder + filename + '.out'
        with open(tag_folder + filename, 'rb') as fp:
            annotations = pickle.load(fp)
        root = ET.parse(file).getroot()
        document = OD() # sent_no: {token_no: features}
        first_sent = True
        coreference = False
        first_cluster = True
        all_entity_coref_clusters = []
        entity_coref_clstr = []
        entity_mention = []
        for element in root.iter():
            if not coreference:
                if element.tag == 'sentence':
                    cur_sent = Sentence(element.attrib['id'])
                    document[element.attrib['id']] = cur_sent
                elif element.tag == 'token':
                    token = Word(element.attrib['id'], filename)
                    cur_sent._add_token(element.attrib['id'], token)
                elif element.tag in ['word', 'lemma', 'CharacterOffsetBegin', 'CharacterOffsetEnd', 'POS', 'NER', 'Speaker']:
                    token._add_feats(element.tag, element.text)
                elif element.tag == 'dependencies' and element.attrib['type'] == 'basic-dependencies':
                    if not first_sent:
                        cur_sent._add_dependency(type, deprels)
                    first_sent = False
                    deprels = []
                    type = 'basic-dependencies'
                elif element.tag == 'dependencies' and element.attrib['type'] == 'collapsed-dependencies':
                    cur_sent._add_dependency(type, deprels)
                    deprels = []
                    type = 'collapsed-dependencies'
                elif element.tag == 'dependencies' and element.attrib['type'] == 'collapsed-ccprocessed-dependencies':
                    cur_sent._add_dependency(type, deprels)
                    deprels = []
                    type = 'collapsed-ccprocessed-dependencies'
                elif element.tag == 'dep':
                    dependency_tuple = [element.attrib['type']]
                elif element.tag == 'governor':
                    dependency_tuple += [element.attrib['idx']]
                elif element.tag == 'dependent':
                    dependency_tuple += [element.attrib['idx']]
                    deprels.append(dependency_tuple)
                elif element.tag == 'coreference':
                    cur_sent._add_dependency(type, deprels)
                    coreference = True
                    # update dependency relations, using collapsed-dependencies
                    for sent_no in document:
                        parse = document[sent_no].collapsed_dependencies
                        for triplets in parse:
                            if triplets[1] == '0':
                                document[sent_no].tokens[triplets[2]].parent_POS = 'ROOT'
                                document[sent_no].tokens[triplets[2]].parent_deprel = triplets[0]
                            else:
                                document[sent_no].tokens[triplets[1]].children_POS.append(document[sent_no].tokens[triplets[2]].POS)
                                document[sent_no].tokens[triplets[1]].children_token_nos.append(document[sent_no].tokens[triplets[2]].token_no)
                                document[sent_no].tokens[triplets[1]].children_deprel.append(triplets[0])
                                document[sent_no].tokens[triplets[2]].parent_POS = document[sent_no].tokens[triplets[1]].POS
                                document[sent_no].tokens[triplets[2]].parent_deprel = triplets[0]
            else:
                if element.tag == 'coreference':
                    if not first_cluster:
                        entity_coref_clstr.append(entity_mention)
                        all_entity_coref_clusters.append(entity_coref_clstr)
                    first_cluster = False
                    entity_coref_clstr = []
                    first_mention = True
                if element.tag == 'mention':
                    if not first_mention:
                        entity_coref_clstr.append(entity_mention)
                    first_mention = False
                    entity_mention = []
                if element.tag in ['sentence', 'start', 'end', 'head']:
                    entity_mention.append(element.text)
        entity_coref_clstr.append(entity_mention)
        all_entity_coref_clusters.append(entity_coref_clstr)
        clstr_id = 121
        for clstr in all_entity_coref_clusters:
            clstr_id += 1
            for clstr_elem in clstr:
                if len(clstr_elem) == 4:
                    document[clstr_elem[0]].tokens[clstr_elem[3]].entity_coref_id = str(clstr_id)
        """
        Below code is for parsing tags
        """
        start_offset_to_em = {}
        for key in annotations:
            if 1:
                start_offset_to_em[annotations[key][0]] = key
            else:
                try:
                    start_offset_to_em[annotations[key][0]] += [key]
                except KeyError: start_offset_to_em[annotations[key][0]] = [key]
        start_offsets = sorted(start_offset_to_em.keys())
        offset_index = 0
        for sent_no in document:
            i_event = False
            for token_no in document[sent_no].tokens:
                if offset_index < len(start_offsets) and document[sent_no].tokens[token_no].CharacterOffsetBegin == start_offsets[offset_index]:
                    offset_key = start_offsets[offset_index]
                    elem = start_offset_to_em[offset_key]
                    if document[sent_no].tokens[token_no].word not in annotations[elem][2]: pass#print document[sent_no].tokens[token_no].word, annotations[elem][2]
                    document[sent_no].tokens[token_no].event_tag = 'B'
                    document[sent_no].tokens[token_no].event_id = elem
                    #print annotations[elem]
                    document[sent_no].tokens[token_no].event_type = annotations[elem][3]
                    document[sent_no].tokens[token_no].event_subtype = annotations[elem][4]
                    document[sent_no].tokens[token_no].event_realis = annotations[elem][5]
                    document[sent_no].tokens[token_no].coref_id = annotations[elem][6]
                    if document[sent_no].tokens[token_no].CharacterOffsetEnd != annotations[elem][1]:
                        i_event = True
                    else:
                        offset_index += 1
                elif i_event:
                    elem = start_offset_to_em[offset_key]
                    document[sent_no].tokens[token_no].event_tag = 'I'
                    document[sent_no].tokens[token_no].event_id = elem
                    #print annotations[elem]
                    document[sent_no].tokens[token_no].event_type = annotations[elem][3]
                    document[sent_no].tokens[token_no].event_subtype = annotations[elem][4]
                    document[sent_no].tokens[token_no].event_realis = annotations[elem][5]
                    document[sent_no].tokens[token_no].coref_id = annotations[elem][6]
                    if document[sent_no].tokens[token_no].CharacterOffsetEnd >= annotations[start_offset_to_em[start_offsets[offset_index]]][1]:
                        offset_index += 1
                        i_event = False
                if not i_event and offset_index < len(start_offsets) and document[sent_no].tokens[token_no].CharacterOffsetBegin > start_offsets[offset_index]:
                    offset_index += 1
                for child in document[sent_no].tokens[token_no].children_token_nos:
                    if document[sent_no].tokens[child].NER in NER_map and document[sent_no].tokens[child].entity_coref_id:
                        document[sent_no].tokens[token_no].args[NER_map[document[sent_no].tokens[child].NER]] += \
                        [document[sent_no].tokens[child].entity_coref_id]

        assert(len(start_offsets) >= offset_index-1) # 1 file (2c7afa4a988d807becade727a769da50) last token anti-pardon

        training_data.append(document)
    if 0:
        for doc in training_data:
            for sent in doc:
                for t in doc[sent].tokens:
                    print t, doc[sent].tokens[t].lemma, doc[sent].tokens[t].args, doc[sent].tokens[t].children_POS, doc[sent].tokens[t].coref_id
        quit()
    return training_data

"""
all_tokens = set()
for doc in training_data:
    for sent_no in doc:
        for token_no in doc[sent_no].tokens:
            all_tokens.add(doc[sent_no].tokens[token_no].word.lower())
            all_tokens.add(doc[sent_no].tokens[token_no].lemma.lower())

print len(all_tokens)
word_vecs = {}
file = open("../../Event_Coref/vocab/glove.840B.300d.txt", 'r')
i = 0
for l in file:
    i += 1
    temp = l.strip().split()
    if temp[0] in all_tokens:
        word_vecs[temp[0]] = [float(elem) for elem in temp[1:]]

with open('../vocab/2016_.pkl', 'wb') as fp:
    pickle.dump(word_vecs, fp)
print i
"""
"""
all_POS = set(['ROOT', 'PADDING'])
for doc in training_data:
    for sent_no in doc:
        for token_no in doc[sent_no].tokens:
            all_POS.add(doc[sent_no].tokens[token_no].POS)

print len(all_POS)
POS_vecs = {}
for index, elem in enumerate(all_POS):
    POS_vecs[elem] = [0.0]*len(all_POS)
    POS_vecs[elem][index] = 1.0
print POS_vecs
with open('../vocab/POS.pkl', 'wb') as fp:
    pickle.dump(POS_vecs, fp)

all_POS = set(['UNKNOWN', 'PADDING'])
for doc in training_data:
    for sent_no in doc:
        for token_no in doc[sent_no].tokens:
            all_POS.add(doc[sent_no].tokens[token_no].parent_deprel)

print len(all_POS)
POS_vecs = {}
for index, elem in enumerate(all_POS):
    POS_vecs[elem] = [0.0]*len(all_POS)
    POS_vecs[elem][index] = 1.0
print POS_vecs
with open('../vocab/deprel.pkl', 'wb') as fp:
    pickle.dump(POS_vecs, fp)
"""
