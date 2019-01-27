import os
import json
import colorlog
import logging
from tqdm import tqdm
import numpy as np
import pickle
import re
from itertools import islice
from collections import Counter
import operator


NUMBER_OF_SENTENCES = 500
MAX_SENTENCE_LEN = 20
DATA_ROOT_PATH = 'instapic'

CAPTION_TRAIN_JSON_FNAME = os.path.join(DATA_ROOT_PATH, 'json', 'insta-caption-train.json')
CAPTION_OUTPUT_PATH = os.path.join(DATA_ROOT_PATH, 'caption_dataset')
VOCAB_FILE = os.path.join(CAPTION_OUTPUT_PATH, 'new.vocab')
WORD2IDX_PICKLE = os.path.join(DATA_ROOT_PATH, 'word_2_idx.pickle')
IDX2WORD_PICKLE = os.path.join(DATA_ROOT_PATH, 'idx_2_word.pickle')
REAL_TEXT = os.path.join(DATA_ROOT_PATH, 'real_data.txt')

# For vocaulary
_PAD = "_pad"
_GO = "_go"
_EOS = "_eos"
_UNK = "_unk"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# For tokenization
try:
    # UCS-4
    EMOTICON = re.compile('(([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF]))')
except Exception as e:
    # UCS-2
    EMOTICON = re.compile('(([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF]))')
NOT_EMOTICON = re.compile(r'(\\U([0-9A-Fa-f]){8})|(\\u([0-9A-Fa-f]){4})')

def load_json(json_fname):
    colorlog.info("Load %s" % (json_fname))
    with open(json_fname, 'r') as f:
        json_object = json.load(f)
    return json_object

def tokenize(sentence):
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)

    sentence = sentence.replace('#', ' #')
    sentence = sentence.replace('@', ' @')
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.lower()
    sentence = re.sub(r"@[a-zA-Z0-9._]+", "@username", sentence)  # change username
    sentence = EMOTICON.sub(r"@@byeongchang\1 ", sentence)
    # sentence = sentence.encode('unicode-escape')  # for emoticons
    sentence = re.sub(r'@@byeongchang\\', '@@byeongchang', sentence)
    sentence = NOT_EMOTICON.sub(r' ', sentence)
    sentence = re.sub(r"[\-_]", r"-", sentence)  # incoporate - and _
    sentence = re.sub(r"([!?,\.\"])", r" ", sentence)  # remove duplicates on . , ! ?
    sentence = re.sub(r"(?<![a-zA-Z0-9])\-(?![a-zA-Z0-9])", r"", sentence)  # remove - if there is no preceed or following
    sentence = ' '.join(re.split(r'[^a-zA-Z0-9#@\'\-]+', sentence))
    sentence = re.sub(r'@@byeongchang', r' \\', sentence)
    return sentence

def tokenize_all(train_json, key='caption'):
    """
    Tokenize sentences in raw dataset
    Args:
    train_json, test1_json, test2_json: raw json object
    key: 'caption' or 'tags'
    """

    colorlog.info("Tokenize %s data" % (key))
    token_counter = Counter()
    train_tokens = {}

    # Train data
    for user_id, posts in tqdm(list(train_json.items()), ncols=70, desc="train data"):
        train_tokens[user_id] = {}
        for post_id, post in list(posts.items()):
            post_tokens = tokenize(post[key])
            post_tokens = post_tokens.split()
            train_tokens[user_id][post_id] = post_tokens
            for post_token in post_tokens:
                token_counter[post_token] += 1

    return token_counter, train_tokens

def pad_sentences(sentence, sentence_len = 20):
    words = [word for word in sentence.split(" ") if word]
    if len(words) > sentence_len:
        words = words[:sentence_len]
    else:
        for i in range(sentence_len-len(words)):
            words.append("_pad")
    return words

def create_vocabulary(counter, fname):
    colorlog.info("Create vocabulary %s" % (fname))
    sorted_tokens = sort_dict(counter)
    vocab = _START_VOCAB + [x[0] for x in sorted_tokens]

    with open(fname, 'w') as f:
        for w in vocab:
            f.write(w + "\n")

    return vocab

def take(n, iterable):
    return dict(islice(iterable, n))

def sort_dict(dic):
    # Sort by alphabet
    sorted_pair_list = sorted(list(dic.items()), key=operator.itemgetter(0))
    # Sort by count
    sorted_pair_list = sorted(sorted_pair_list, key=operator.itemgetter(1), reverse=True)
    return sorted_pair_list

def vocab_mapping():
    colorlog.info("Pickling word to index and index to word mappings")
    word_to_idx = {}
    counter = 0

    with open(VOCAB_FILE, "r") as f:
        word_list = f.readlines()
        for word in word_list:
            word = word.strip()
            word = word.replace("\n","")
            if word not in word_to_idx:
                word_to_idx[word] = counter
                counter += 1

    idx_to_word = {idx: word for idx, word in enumerate(word_to_idx)}

    # with open(WORD2IDX_PICKLE, "wb") as p:
    #     pickle.dump(word_to_idx, p)
    # with open(IDX2WORD_PICKLE, "wb") as p:
    #     pickle.dump(idx_to_word, p)

    return word_to_idx, idx_to_word

def main():
    colorlog.basicConfig(
      filename=None,
      level=logging.INFO,
      format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
    )

    # Load raw data
    caption_train_json = load_json(CAPTION_TRAIN_JSON_FNAME)
    caption_train_json_1 = take(200, caption_train_json.items())

    caption_counter, caption_train_tokens = tokenize_all(caption_train_json_1, 'caption')
    caption_vocab = create_vocabulary(caption_counter, VOCAB_FILE)

    word_2_idx, idx_2_word = vocab_mapping()
    post_caption_dict = {}
    i = 0
    for user_id, posts in caption_train_json_1.items():
        if i < 1:
            j = 0
            for post_id, post in list(posts.items()):
                if j < NUMBER_OF_SENTENCES:
                    if post['m_id'] not in post_caption_dict:
                        post_tokens = tokenize(post['caption'])
                        post_caption_dict[post['m_id']] = post_tokens
                j += 1
        i += 1

    out_file = open(REAL_TEXT, "w")
    for _, post in post_caption_dict.items():
        post = pad_sentences(post)
        sep = ''
        for word in post:
            out_file.write(sep + str(word_2_idx[word.lower()]+1))
            sep = ' '
        out_file.write('\n')

    out_file.close()

if __name__ == '__main__':
    main()
