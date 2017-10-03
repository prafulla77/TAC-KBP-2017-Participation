'''
Parse ERE documents
Only care about event hoppers
'''

import xml.etree.ElementTree as ET
import os, codecs, re, pickle

def getCleanSentence(line):
    anno = re.compile("<[^>]*>")
    url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    xml_clean = anno.sub('', line)
    clean = url.sub('weblink', xml_clean)
    return clean.replace('\n', ' ')

filenames = os.listdir('../data/2015/training/source')
file_list = open('../data/2015/training/files.txt', 'w')
for filename in filenames:
    offset2tag = {}
    ev_id2attrib = {}
    hoppers = ET.parse('../data/2015/training/event_hopper/' + filename[:-4] + '.event_hoppers.xml').getroot().find('hoppers')
    with codecs.open('../data/2015/training/source/' + filename, 'r', encoding='utf8') as content_file:
        source = content_file.read()

    offsets = []
    for hopper in hoppers.findall('hopper'):
        h_id = hopper.attrib['id']
        for event_mention in hopper.findall('event_mention'):
            start = int(event_mention.find('trigger').attrib['offset'])
            end = start + int(event_mention.find('trigger').attrib['length'])
            #event_word = event_mention.find('trigger').text
            ev_id2attrib[event_mention.attrib['id']] = [start, end, event_mention.find('trigger').text, event_mention.attrib['type'], event_mention.attrib['subtype'], event_mention.attrib['realis'], h_id]
            offsets += [start, end]
    offsets.sort()
    prev = 0
    clean_source = ''
    for curr in offsets:
        clean_source += getCleanSentence(source[prev:curr])
        offset2tag[curr] = len(clean_source)
        prev = curr
        #source = source[:key] + offset2tag[key] + source[key:]

    for key in ev_id2attrib:
        if clean_source[offset2tag[ev_id2attrib[key][0]]:offset2tag[ev_id2attrib[key][1]]] == ev_id2attrib[key][2]:
            ev_id2attrib[key][0] = offset2tag[ev_id2attrib[key][0]]
            ev_id2attrib[key][1] = offset2tag[ev_id2attrib[key][1]]

    with open('../data/2015/training/event_tags/'+filename, 'wb') as destination_file:
        pickle.dump(ev_id2attrib, destination_file)

    with codecs.open('../data/2015/training/clean_source/'+filename, 'w', encoding='utf8') as destination_file:
        destination_file.write(clean_source)

    file_list.write('/Users/prafulla/Desktop/Event_Trigger/data/2015/training/clean_source/'+filename+'\n')