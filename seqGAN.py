import numpy as np
import tensorflow as tf
import random
from data_loader import Discriminator_Data_Loader, Generator_Data_Loader
from generator import Generator
from discriminator import Discriminator
from target_lstm import TARGET_LSTM
from rollout import ROLLOUT
import pickle

tf.reset_default_graph()
# Generator Hyper Parameters
# EMB_DIM - embedding dimension
# HIDDEN_DIM - hidden state dimension of lstm cell
# SEQ_LENGTH - sequence length
# PRE_EPOCH_NUM - supervise (maximum likelihood estimation) epochs
EMB_DIM = 32
HIDDEN_DIM = 32
SEQ_LENGTH = 20
START_TOKEN = 0
PRE_EPOCH_NUM = 120
SEED = 88
BATCH_SIZE = 64

# Discriminator Hyper Parameters
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
# dis_batch_size = 64

# Basic Training Parameters
TOTAL_BATCH = 200
positive_file = 'data/real_data.txt'
negative_file = 'data/generator_sample.txt'
eval_file = 'data/eval_file.txt'
generated_num = 10000

# Generate data samples - will use Generator model
def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as f:
        for sample in generated_samples:
            buffer = ' '.join([str(x) for x in sample]) + '\n'
            f.write(buffer)

# target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
# For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
def target_loss(sess, target_lstm, data_loader):
    nll = []
    data_loader.reset_pointer()

    for _ in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)

# Pre-train the generator using MLE for one epoch
def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for _ in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Generator_Data_Loader(BATCH_SIZE)
    # For testing
    likelihood_data_loader = Generator_Data_Loader(BATCH_SIZE)
    vocab_size = 5000
    dis_data_loader = Discriminator_Data_Loader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open('data/target_params_py3.pkl', 'rb'))
    # The oracle model - synthetic data
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params)

    discriminator = Discriminator(seq_len=20, num_classes=2, vocab_size=vocab_size, emb_size=dis_embedding_dim, filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('data/experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('Pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print(f'Pre-train epoch: {epoch}, Test_loss: {test_loss}')
            buffer = "Epoch:\t"+ str(epoch) + "\tNeg-Log Likelihood:\t" + str(test_loss) + "\n"
            log.write(buffer)

    print('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(50):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('Adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = "Epoch:\t" + str(total_batch) + "\tNegative Log Likelihood:\t" + str(test_loss) + "\n"
            print(f'Total Batch: {total_batch}, Test Loss {test_loss}')
            log.write(buffer)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

    log.close()

if __name__ == '__main__':
    main()
