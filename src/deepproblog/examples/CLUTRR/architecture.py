from random import choices

import torch
import torch.nn as nn
from typing import List
from problog.logic import term2list


def tokenize2(text, entities, vocab):
    text = text.strip('"').split(" ")
    tokenized = []
    for i, word in enumerate(text):
        try:
            word = int(word)
            try:
                word = entities.index(word) + 1
            except ValueError:
                word = 0

            tokenized.append(word)

        except ValueError:
            try:
                tokenized.append(vocab[word])
            except KeyError:
                tokenized.append(vocab["OOV"])
    return tokenized


def tokenize(text, entity, vocab):
    text = text.strip('"').split(" ")
    tokenized = []
    indices = []
    for i, word in enumerate(text):
        try:
            word = int(word)
            if word == entity:
                word = 1
                indices.append(i)
            else:
                word = 0

            tokenized.append(word)

        except ValueError:
            try:
                tokenized.append(vocab[word])
            except KeyError:
                tokenized.append(vocab["OOV"])
    return tokenized, indices


def tokenize_cloze(text, entities, vocab, nr_entities=10):
    text = text.strip('"').split(" ")
    entity_tokens = ["ENT{}".format(x) for x in range(nr_entities)]
    entity_tokens = choices(entity_tokens, k=len(entities))
    tokenized = []
    indices: List[List[int]] = [[] for _ in range(len(entities))]
    for i, word in enumerate(text):
        try:
            word = int(word)
            indices[entities.index(word)].append(i)
            word = vocab[entity_tokens[entities.index(word)]]
            tokenized.append(word)
        except ValueError:
            try:
                tokenized.append(vocab[word])
            except KeyError:
                tokenized.append(vocab["OOV"])
    return tokenized, indices


class Encoder(nn.Module):
    def __init__(self, vocab, hidden_size, embed_size=None, p_drop=0.0, weights=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        if embed_size is None:
            if weights is None:
                self.embed_size = hidden_size
            else:
                self.embed_size = weights.shape[1]
        else:
            self.embed_size = embed_size
        self.vocab = vocab
        self.lstm = nn.GRU(
            self.embed_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=p_drop,
        )
        self.dropout = nn.Dropout(p_drop)
        if weights is None:
            self.embedding = nn.Embedding(len(self.vocab), self.embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(weights)

    def forward(self, text, ent1, ent2):
        text = text.value.strip('"')
        ent1, ent2 = int(ent1), int(ent2)
        x, indices = tokenize_cloze(text, [ent1, ent2], self.vocab)
        x = self.embedding(torch.LongTensor(x))
        x, _ = self.lstm(x.unsqueeze(0))
        x = x.view(-1, 2, self.hidden_size)
        forward, backward = x[:, 0, :], x[:, 1, :]
        out = []
        for j, e in enumerate([ent1, ent2]):
            i = indices[j]
            if i:
                i = torch.LongTensor(i)
                embedding = torch.cat([forward[i, :], backward[-i - 1, :]])
                out.append(torch.mean(embedding, 0))
            else:
                out.append(None)
        return out


class RelNet(nn.Module):
    def __init__(self, in_size, mid_size, out_size=11, activation=True):
        super(RelNet, self).__init__()
        self.in_size = in_size
        self.mid_size = mid_size
        self.out_size = in_size if out_size is None else out_size
        self.embed = nn.Sequential(nn.Linear(2 * self.in_size, self.out_size), )
        self.activation_layer = nn.Softmax(dim=-1)
        self.activation = activation

    def forward(self, ex, ey):
        x = torch.cat([ex, ey], 0)
        x = self.embed(x.unsqueeze(0))
        if self.activation:
            x = self.activation_layer(x)
        return x[0]


class GenderNet(nn.Module):
    def __init__(self, vocab, hidden_size, embed_size=None):
        super(GenderNet, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        if embed_size is None:
            self.embed_size = hidden_size

        self.vocab = vocab
        self.lstm = nn.GRU(
            self.embed_size,
            hidden_size,
            bidirectional=False,
            batch_first=True,
            num_layers=1,
        )
        self.embedding = nn.Embedding(len(self.vocab), self.embed_size)
        self.classification = nn.Sequential(
            nn.Linear(self.hidden_size, 2), nn.Softmax(-1)
        )

    def forward(self, text, ent):
        text = [x.args[1] for x in term2list(text, False)]
        text = " ".join(t.value.strip('"') + " ." for t in text)
        ent = int(ent)
        x, indices = tokenize_cloze(text, [ent], self.vocab)
        x = self.embedding(torch.LongTensor(x))
        x, _ = self.lstm(x.unsqueeze(0))
        x = x[:, -1, :]
        x = self.classification(x)
        return x[0]
