# SeqGAN
Tensorflow implementation for the paper SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient (https://arxiv.org/pdf/1609.05473.pdf) by Lantao Yu et al.
This repository is created to implement the architecture of the paper on synthetic data(like in paper) and any text dataset available publicly. <br>

- data_loader.py is responsible for loading data in batches for both, Generator and Discriminator.
- discriminator.py has the architecture for Discriminator model, which is a Convolutional Neural Network for text classification. It also uses a highway network.
- generator.py has the architecture for Generator model, which according to the paper is a Recurrent Neural Network with LSTM units.
- target_lstm.py is similar to generator and responsible for creating synthetic data. It can be omitted if tested on a dataset.

I have used InstaPic dataset captions to generate new set of captions from SeqGAN along with the synthetic dataset procedure the authors have presented in the paper.
