import xml.etree.ElementTree as ET
import os, codecs, re, pickle

def getCleanSentence(line):
    anno = re.compile("<[^>]*>")
    url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    xml_clean = anno.sub('', line)
    clean = url.sub('weblink', xml_clean)
    return clean.replace('\n', ' ')

file_list = open('../data/2016/files.txt', 'w')
for df_nw in ['df', 'nw']:
    filenames = os.listdir('../data/2016/' + df_nw +'/source')
    for filename in filenames:
        offset2tag = {}
        ev_id2attrib = {}
        hoppers = ET.parse('../data/2016/' + df_nw +'/ere/' + filename[:-4] + '.rich_ere.xml').getroot().find('hoppers')
        with codecs.open('../data/2016/' + df_nw +'/source/' + filename, 'r', encoding='utf8') as content_file:
            source = content_file.read()

        offsets = []
        for hopper in hoppers.findall('hopper'):
            h_id = hopper.attrib['id']
            for event_mention in hopper.findall('event_mention'):
                start = int(event_mention.find('trigger').attrib['offset'])
                end = start + int(event_mention.find('trigger').attrib['length'])
                ev_id2attrib[event_mention.attrib['id']] = [start, end, event_mention.find('trigger').text, event_mention.attrib['type'], event_mention.attrib['subtype'], event_mention.attrib['realis']]
                offsets += [start, end]
        offsets.sort()
        prev = 0
        clean_source = ''
        for curr in offsets:
            clean_source += getCleanSentence(source[prev:curr])
            offset2tag[curr] = len(clean_source)
            prev = curr

        for key in ev_id2attrib:
            if clean_source[offset2tag[ev_id2attrib[key][0]]:offset2tag[ev_id2attrib[key][1]]] == ev_id2attrib[key][2]:
                ev_id2attrib[key][0] = offset2tag[ev_id2attrib[key][0]]
                ev_id2attrib[key][1] = offset2tag[ev_id2attrib[key][1]]

        with open('../data/2016/event_tags/'+filename, 'wb') as destination_file:
            pickle.dump(ev_id2attrib, destination_file)

        with codecs.open('../data/2016/clean_source/'+filename, 'w', encoding='utf8') as destination_file:
            destination_file.write(clean_source)

        file_list.write('/Users/prafulla/Desktop/Event_Trigger/data/2016/clean_source/'+filename+'\n')