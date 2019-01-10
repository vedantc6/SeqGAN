import numpy as np

# Data loader for Generator
class Generator_Data_Loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.word_stream = []

    def create_batches(self, data_file):
        self.word_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                line_list = [int(x) for x in line]
                # One can do padding or trim the line to length of 20
                if len(line_list) == 20:
                    self.word_stream.append(line_list)

        self.num_batch = int(len(self.word_stream)/self.batch_size)
        self.word_stream = self.word_stream[:self.num_batch*self.batch_size]
        self.sequence_batch = np.split(np.array(self.word_stream), self.num_batch, axis=0)
        self.pointer = 0

    def next_batch(self):
        retrieve = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return retrieve

    def reset_pointer(self):
        self.pointer = 0

# Data loader for Discriminator
class Discriminator_Data_Loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, pos_file, neg_file):
        pos_examples = []
        neg_examples = []

        with open(pos_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                line_list = [int(x) for x in line]
                pos_examples.append(line_list)

        with open(neg_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                line_list = [int(x) for x in line]
                if len(line_list) == 20:
                    neg_examples.append(line_list)

        self.sentences = np.array(pos_examples + neg_examples)

        # Generate labels
        pos_labels = [[0, 1] for _ in pos_examples]
        neg_labels = [[1, 0] for _ in neg_examples]

        self.labels = np.concatenate([pos_labels, neg_labels], 0)

        # Shuffle the data on shuffle_indexes
        shuffle_ind = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_ind]
        self.labels = self.labels[shuffle_ind]

        # Split into batches
        self.num_batch = int(len(self.labels)/self.batch_size)
        self.sentences = self.sentences[:self.num_batch*self.batch_size]
        self.labels = self.labels[:self.num_batch*self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, axis=0)
        self.labels_batches = np.split(self.labels, self.num_batch, axis=0)

        self.pointer = 0

    def next_batch(self):
        retrieve = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return retrieve

    def reset_pointer(self):
        self.pointer = 0
