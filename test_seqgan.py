import os
import colorlog
DATA_ROOT_PATH = 'data'
INSTA_DATA_PATH = 'instapic'
RESULT_TEXT = os.path.join(DATA_ROOT_PATH, 'final.txt')
CAPTION_OUTPUT_PATH = os.path.join(INSTA_DATA_PATH, 'caption_dataset')
VOCAB_FILE = os.path.join(CAPTION_OUTPUT_PATH, 'new.vocab')

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
    return word_to_idx, idx_to_word

with open(RESULT_TEXT, "r") as f:
    results = f.readlines()

results = [x.strip() for x in results]
_, idx_to_word = vocab_mapping()
for result in results[:5]:
    nums = result.split()
    sentence = ''
    for num in nums:
        sentence += idx_to_word[int(num)-1] + ' '
    print(sentence)
    print("\n")
