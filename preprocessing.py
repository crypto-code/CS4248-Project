from fileinput import filename
from msilib.schema import Class
import os
import re
import json
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

    # def lemmatize_word(self, word):
    #     lemmatizer = WordNetLemmatizer()
    #     result = re.findall('\w+', word)
    #     # print(result)
    #     for r in result:
    #         lemmatized = lemmatizer.lemmatize(r)
    #         if r != lemmatized:
    #             word = word.replace(r, lemmatized)
    #     return word

    # def lemmatize_phrase(self, text):
    #     return ' '.join(map(lambda w: self.lemmatize_word(w), [w for w in text.split(' ')]))

    # def find_all_substring_pos(self, text, to_find):
    #     return [m.start() for m in re.finditer(to_find, text)]

    # text is lemmatized
    # kw is lemmatized
    # array of kw is ordered according start/end idx

    # def fix_kw(self, text, kw, array_of_kw):
    #     arr_start_idx = self.find_all_substring_pos(text, kw)
    #     # assumption: len of start idx == len of array of kw
    #     fixed = []
    #     for idx, v in enumerate(array_of_kw):
    #         _, _, t = v
    #         fixed.append([arr_start_idx[idx], arr_start_idx[idx] + len(kw), t])
    #     return fixed

    def lemmatize(self, text):
        self.doc = self.nlp(text)
        lem_text = ' '.join(map(lambda x: x.lemma_, self.doc))
        return lem_text

    def lemmatize_kw_dict(self, lem_text, word_kw_dict):
        lem_word_offsets = dict()
        lem_char_offsets = dict()
        lem_split = lem_text.split(' ')
        lem_char_counts = [0]
        char_index = 0

        # get the character offsets for the beginning of each word
        for lem in lem_split:
            char_index += len(lem) + 1
            lem_char_counts.append(char_index)

        # map the word offsets and character offsets to each word
        for _, value in word_kw_dict.items():
            for v in value:
                try:
                    key = ' '.join(lem_split[v[0]: v[1]])
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

    # 'This is a bird'
    # char index array  [0,0,0,0,0,  1,1,1,  2,2,  3,3,3,3,3]
    def to_word_index(self, text, kw_dict):
        doc = self.nlp(text)            # tokenize text
        str_tokens = list(map(lambda x: str(x), doc))
        print(str_tokens)
        char_to_word_map = []
        curr_word = ''
        curr_word_index = 0

        for c in text:
            # space or newline is included as part of a word
            if c == ' ' or c == '\n':
                char_to_word_map.append(curr_word_index)
                curr_word = ''
                continue
            curr_word += c

            if curr_word == str_tokens[curr_word_index]:
                char_to_word_map.extend([curr_word_index for k in curr_word])
                curr_word = ''
                curr_word_index += 1
        if len(char_to_word_map) != len(text):
            raise Exception(
                f'len char map: {len(char_to_word_map)} len text: {len(text)}\n{text}\n{str_tokens}\n{char_to_word_map}')
        # print(char_to_word_map)
        new_dict = dict()
        # To check: key is not in tokenized form yet in new_dict
        # 'ni-usb 6009 analog-to-digital converter': [[77, 87, 'Material']]
        # tokenized form: 'ni - usb 6009 analog - to - digital converter'
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
        # lem_text = self.lemmatize_phrase(text)
        # lem_kw_dict = dict()
        # for kw, arr_kw in kw_dict.items():
        #     lem_kw = self.lemmatize_phrase(kw)
        #     fixed = self.fix_kw(lem_text, lem_kw, arr_kw)
        #     lem_kw_dict[lem_kw] = fixed

        # word_kw: {'value of the measuring resistor': [[7, 12, 'Process']], 'rm': [[13, 14, 'Process']], ... }
        word_kw_dict = self.to_word_index(text, kw_dict)
        # print(word_kw_dict)
        # the act of lemmatizing is adding additional space to a space token
        lem_text = self.lemmatize(text)
        doc = self.nlp(text)            # tokenize text
        str_tokens = list(map(lambda x: str(x), doc))
        print(str_tokens)
        # self.doc = self.nlp(text)
        # lem_text = ' '.join(map(lambda x: x.lemma_, self.doc))
        print(lem_text)
        print(" ".join(lem_text.split()))
        lem_kw_char_offsets, lem_kw_word_offsets = self.lemmatize_kw_dict(
            lem_text, word_kw_dict)
        # print(lem_kw_char_offsets)
        self.json_file[filename] = dict()
        self.json_file[filename]['lem_text'] = lem_text
        self.json_file[filename]['lem_keywords_word_offset'] = lem_kw_word_offsets
        self.json_file[filename]['lem_keywords_char_offset'] = lem_kw_char_offsets
        # print(self.json_file[filename]['lem_keywords'])

    def get_text(self, text, filename):
        self.json_file[filename] = dict()
        self.json_file[filename]['text'] = text

    def get_kw(self, kw_dict, filename):
        self.json_file[filename]['keywords'] = kw_dict

    def tokenize(self, text, filename):
        tokens = word_tokenize(text)
        self.json_file[filename]['word_tokens'] = tokens

    def pos(self, text, filename):
        # pos = pos_tag(word_tokenize(text))
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
        # should we ner on the lowercased text?
        # eg. Apple (company) vs apple (fruit) will be recognised differently
        # [('9', 'CARDINAL'), ('4.7kω', 'CARDINAL'), ('a ni-usb', 'DATE'), ('1000', 'CARDINAL'),
        # ('1000', 'CARDINAL'), ('1±0.05s', 'CARDINAL'), ('between 0.5 and', 'CARDINAL'), ('50hz', 'ORDINAL')]
        # ?: not sure why ner is inputing cardinal and ordinal only
        doc = self.nlp(text)
        # print(doc)
        ner = []
        for ent in doc.ents:
            ner.append((ent.text, ent.label_))
        self.json_file[filename]['entities'] = ner

    def readAnn(self, textfolder="scienceie2017_dev/dev/"):
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
                # print(anno_inst)
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
                        # kw_dict_lem[lemmatizer.lemmatize(anno_inst[2])] = [
                        #    start, end, keytype]
                        # look up span in text and print error message if it doesn't match the .ann span text
                        # keyphr_text_lookup = text[int(start):int(end)]
                        # print('span')
                        # print(f'keyphr lookup:{keyphr_text_lookup}')
                        # keyphr_ann = anno_inst[2]
                        # print(keyphr_ann)
                        # if keyphr_text_lookup != keyphr_ann:

                            # print("Spans don't match for anno " +
                            #      l.strip() + " in file " + f)
            # print(f'kw dict: {kw_dict}')
            # print(f'after lemmatization: {kw_dict_lem}')
            normalized_text = re.sub(
                ' +', ' ', unicodedata.normalize("NFKD", text))

            print("hi")
            print(normalized_text)
            self.lematize_text_and_kw(normalized_text, kw_dict, f)

            # self.get_text(text, f)
            # self.get_kw(kw_dict, f)

            self.tokenize(text, f)
            self.sent_tokenize(text, f)
            self.pos(text, f)
            self.chunk(text, f)
            self.name_entity_recognition(text, f)
            json_file = json.dumps(self.json_file, indent=4)
            with open('json_data.json', 'w') as outfile:
                outfile.write(json_file)
            # print(self.json_file[f]['text'])
            # print(json_file)


scienceIE = ScienceIE()

scienceIE.readAnn()
