import torch
import torch.nn as nn
import torch.optim as optim

from deepproblog.utils import count_parameters
from deepproblog.utils.standard_networks import MLP

vocab = dict()
hidden_size = 512
embed_size = 256

with open("data/vocab_746.txt") as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i


def tokenize(sentence):
    sentence = sentence.split(" ")
    tokens = []
    numbers = list()
    indices = list()
    for i, word in enumerate(sentence):
        if word.isdigit():
            numbers.append(int(word))
            tokens.append("<NR>")
            indices.append(i)
        else:
            if word in vocab:
                tokens.append(word)
            else:
                tokens.append("<UNK>")
    return [vocab[token] for token in tokens], numbers, indices


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(RNN, self).__init__()
        self.lstm = nn.GRU(
            embed_size, hidden_size, 1, bidirectional=True, batch_first=True
        )
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            vocab_size, embed_size
        )  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)

    def forward(self, sentence):
        sentence = sentence.value.strip('"')
        x, _, indices = tokenize(sentence)
        n1, n2, n3 = indices
        seq_len = len(x)
        x = torch.LongTensor(x).unsqueeze(0)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(seq_len, 2, self.hidden_size)
        x1 = torch.cat([x[-1, 0, ...], x[n1, 0, ...], x[n2, 0, ...], x[n3, 0, ...]])
        x2 = torch.cat([x[0, 1, ...], x[n1, 1, ...], x[n2, 1, ...], x[n3, 1, ...]])
        x = torch.cat([x1, x2])
        #        return x
        return self.dropout(x)


def get_networks(lr, p_drop, n=8):
    rnn = RNN(len(vocab), hidden_size, p_drop=p_drop)
    network1 = MLP(hidden_size * n, 6)
    network2 = MLP(hidden_size * n, 4)
    network3 = MLP(hidden_size * n, 2)
    network4 = MLP(hidden_size * n, 4)

    print("rnn params", count_parameters(rnn))
    print("net1 params", count_parameters(network1))

    names = ["rnn", "nn_permute", "nn_op1", "nn_swap", "nn_op2"]
    networks = [rnn, network1, network2, network3, network4]

    networks = [
        (networks[i], names[i], optim.Adam(networks[i].parameters(), lr=lr))
        for i in range(5)
    ]

    return networks
