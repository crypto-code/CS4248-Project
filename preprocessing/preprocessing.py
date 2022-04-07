from fileinput import filename
from msilib.schema import Class
import os
import re
import json
from turtle import st
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, sent_tokenize
from numpy import unicode_
import spacy


class ScienceIE:
    def __init__(self):
        self.json_file = dict()
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = None

    def lemmatize(self, text):
        self.doc = self.nlp(text)
        lem_list = list(map(lambda x: x.lemma_, self.doc))

        lem_text = ' '.join(map(lambda x: x.lemma_, self.doc))
        return lem_list, lem_text

    def lemmatize_kw_dict(self, text, word_kw_dict):
        lem_word_offsets = dict()
        lem_char_offsets = dict()
        self.doc = self.nlp(text)
        lem_list = list(map(lambda x: x.lemma_, self.doc))
        lem_char_counts = [0]
        char_index = 0

        # get the character offsets for the beginning of each word
        for lem in lem_list:
            char_index += len(lem) + 1
            lem_char_counts.append(char_index)

        # map the word offsets and character offsets to each word
        for _, value in word_kw_dict.items():
            for v in value:
                try:
                    key = ' '.join(lem_list[v[0]: v[1]])
                    if key not in lem_word_offsets:
                        lem_word_offsets[key] = []
                        lem_char_offsets[key] = []
                    # 7 12
                    lem_word_offsets[key].append([v[0], v[1], v[2]])
                    lem_char_offsets[key].append(
                        [lem_char_counts[v[0]], lem_char_counts[v[1]] - 1, v[2]])
                except:
                    print(v)
                    raise

        return lem_char_offsets, lem_word_offsets

    # Eg. 'This is a bird'
    # char index array  [0,0,0,0,0,  1,1,1,  2,2,  3,3,3,3,3]
    def to_word_index(self, text, kw_dict):
        doc = self.nlp(text)            # tokenize text
        str_tokens = list(map(lambda x: str(x), doc))
        char_to_word_map = []
        curr_word = ''
        curr_word_index = 0

        for c in text:
            # space or newline is included as part of a word unless it is a token by itself
            if c == ' ' or c == '\n':
                if c == str_tokens[curr_word_index] or c + ' ' == str_tokens[curr_word_index]:
                    curr_word_index += 1

                char_to_word_map.append(curr_word_index)
                curr_word = ''
                continue
            curr_word += c

            if curr_word == str_tokens[curr_word_index]:
                char_to_word_map.extend([curr_word_index for k in curr_word])
                curr_word_index += 1
                curr_word = ''
        if len(char_to_word_map) != len(text):
            raise Exception(
                f'len char map: {len(char_to_word_map)} len text: {len(text)}\n{text}\n{str_tokens}\n{char_to_word_map}')
        new_dict = dict()
        for key, value in kw_dict.items():
            new_dict[key] = []
            for v in value:
                try:
                    new_dict[key].append([char_to_word_map[v[0]],
                                          char_to_word_map[v[1]], v[2]])
                except:
                    print(v)
                    print(len(char_to_word_map))
                    raise
        return new_dict

    def lematize_text_and_kw(self, text, kw_dict, filename):

        word_kw_dict = self.to_word_index(text, kw_dict)

        lem_list, lem_text = self.lemmatize(text)
        lem_kw_char_offsets, lem_kw_word_offsets = self.lemmatize_kw_dict(
            text, word_kw_dict)
        self.json_file[filename] = dict()
        self.json_file[filename]['lem_text'] = lem_text
        self.json_file[filename]['lem_list'] = lem_list
        self.json_file[filename]['lem_keywords_word_offset'] = lem_kw_word_offsets
        self.json_file[filename]['lem_keywords_char_offset'] = lem_kw_char_offsets

    def get_text(self, text, filename):
        self.json_file[filename]['text'] = text

    def get_kw(self, kw_dict, filename):
        self.json_file[filename]['keywords'] = kw_dict

    def tokenize(self, text, filename):
        doc = self.nlp(text)
        tokens = list(map(lambda x: str(x), doc))
        self.json_file[filename]['word_tokens'] = tokens

    def pos(self, text, filename):
        if self.doc == None:
            self.lemmatize(text)
        pos = list(map(lambda x: [x.lemma_, x.tag_], self.doc))
        self.json_file[filename]['pos'] = pos

    def sent_tokenize(self, text, filename):
        sent_tokens = sent_tokenize(text)
        self.json_file[filename]['sent_tokens'] = sent_tokens

    def chunk(self, text, filename):
        doc = self.nlp(text)
        chunks = []
        for chunk in doc.noun_chunks:
            chunks.append(chunk.text)
        self.json_file[filename]['chunks'] = chunks

    def name_entity_recognition(self, text, filename):
        doc = self.nlp(text)
        ner = []
        for ent in doc.ents:
            ner.append((ent.text, ent.label_))
        self.json_file[filename]['entities'] = ner

    def readAnn(self, textfolder="scienceie2017_train/train2/"):
        '''
        Read .ann files and look up corresponding spans in .txt files
        :param textfolder:
        :return:
        '''

        flist = os.listdir(textfolder)
        for f in flist:
            if not f.endswith(".ann"):
                continue
            f_anno = open(os.path.join(textfolder, f), "r", encoding='utf-8')
            f_text = open(os.path.join(
                textfolder, f.replace(".ann", ".txt")), "r", encoding='utf-8')

            # there's only one line, as each .ann file is one text paragraph
            text = f_text.read().lower()
            for l in f_text:
                text = l
            kw_dict = dict()
            for l in f_anno:
                anno_inst = l.strip("\n").split("\t")
                if len(anno_inst) == 3:
                    anno_inst1 = anno_inst[1].split(" ")
                    if len(anno_inst1) == 3:
                        keytype, start, end = anno_inst1
                    else:
                        keytype, start, _, end = anno_inst1
                    if not keytype.endswith("-of"):
                        key = anno_inst[2].lower()
                        if key not in kw_dict:
                            kw_dict[key] = [[int(start), int(end), keytype]]
                        else:
                            # kw_dict --> eg. { "value of the measuring resistor" : [[39, 70, 'Process']] }
                            kw_dict[key].append(
                                [int(start), int(end), keytype])

            normalized_text = unicodedata.normalize("NFKD", text)
            dp_whitespace_text = re.sub(' +', ' ', normalized_text)
            self.lematize_text_and_kw(normalized_text, kw_dict, f)

            self.get_text(text, f)
            self.get_kw(kw_dict, f)

            self.tokenize(text, f)
            self.sent_tokenize(text, f)
            self.pos(text, f)
            self.chunk(text, f)
            self.name_entity_recognition(text, f)
            print("len json file", len(self.json_file))
            json_file = json.dumps(self.json_file, indent=4)
            with open('json_data_train.json', 'w') as outfile:
                outfile.write(json_file)


scienceIE = ScienceIE()

scienceIE.readAnn()
