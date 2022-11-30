import csv
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from random import choices
from re import match

import torch
from torch.utils.data import Dataset as PyDataset

from typing import Union
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant, list2term

# 1 noise_free
# 2 supporting
# 3 irrelevant
# 4 disconnected

dataset_names = {
    "noise_0": "train 1.2, 1.3, test 1.2,1.3,2.3,3.3,4.3",
    "noise_1": "train 2.2, 2.3, test 1.3,2.2,2.3,3.3,4.3",
    "noise_2": "train 3.2, 3.3, test 1.3,2.3,3.2,3.3,4.3",
    "noise_3": "train 4.2, 4.3, test 1.3,2.3,3.3,4.2,4.3",
    "sys_gen_0": "train 1.2 test 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10",
    "sys_gen_1": "train 1.2,1.3 test 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10",
    "sys_gen_2": "train 1.2,1.3,1.4 test 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10",
}

synonyms = {
    "sys_gen_0": "data/data_a9dcffad",
    "sys_gen_1": "data/data_b11e6a4f/",
    "sys_gen_2": "data/data_f70b574f",
    "noise_0": "data/data_2d5007e7/",
    "noise_1": "data/data_a7d9402e/",
    "noise_2": "data/data_6b1c2f15/",
    "noise_3": "data/data_47b0ffea/",
}

labels1 = [
    "daughter",
    "daughter_in_law",
    "mother",
    "mother_in_law",
    "sister",
    "sister_in_law",
    "grandmother",
    "granddaughter",
    "niece",
    "aunt",
    "wife",
]
labels2 = [
    "son",
    "son_in_law",
    "father",
    "father_in_law",
    "brother",
    "brother_in_law",
    "grandfather",
    "grandson",
    "nephew",
    "uncle",
    "husband",
]
labels3 = [
    "child",
    "child_in_law",
    "parent",
    "parent_in_law",
    "sibling",
    "sibling_in_law",
    "grandparent",
    "grandchild",
    "nephew",
    "uncle",
    "so",
]

labels = labels1 + labels2


def remove_gender(x):
    if x in labels3:
        return x
    try:
        return labels3[labels1.index(x)]
    except ValueError:
        return labels3[labels2.index(x)]


class ListDataset(PyDataset):
    def __init__(self, data):
        PyDataset.__init__(self)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def parse_relation(x):
    x = x.replace("-", "_")
    if x == "neice":
        x = "niece"
    return x


class Story(object):
    def __init__(self, line=None, vocab=None, copy=None, edges=True):
        self.use_edges = edges
        self.positive = True
        if copy is not None:
            self.edges = copy.edges
            self.query = copy.query
            self.answer = copy.answer
            self.text = copy.text
            self.entities = copy.entities
        else:
            text, query, answer = line[2], line[3], line[5]
            story_edges, edge_types, genders = line[11], line[12], line[14]
            story_edges = literal_eval(story_edges)
            edge_types = literal_eval(edge_types)
            edge_types = [parse_relation(x) for x in edge_types]
            try:
                self.edges = {
                    story_edges[m]: edge_types[m] for m in range(len(story_edges))
                }
            except IndexError:
                self.edges = {}

            self.query = tuple(line.strip("'").lower() for line in query[1:-1].split(", "))
            answer = parse_relation(answer)
            if answer not in labels:
                raise Exception(answer + " not in labels")
            self.answer = answer

            genders = [g.split(":")[0].lower() for g in genders.strip().split(",")]

            for punctuation in [".", ",", "'", "!", ";"]:
                text = text.replace(punctuation, " " + punctuation + " ")
            self.text = [w for w in text.strip().split(" ") if len(w) > 0]

            self.entities = {name: ind for ind, name in enumerate(genders)}
            for i in range(len(self.text)):
                if self.text[i][0] == "[":
                    word = self.text[i][1:-1].lower()
                    j = self.entities[word]
                    self.text[i] = j
                else:
                    self.text[i] = self.text[i].lower()
                    word = self.text[i].lower()
                    vocab[word] += 1

    def get_sentences(self):
        sentences = [[]]
        entities = [set()]
        for word in self.text:
            if word in [".", "!"]:
                sentences.append([])
                entities.append(set())
            else:
                if type(word) is int:
                    entities[-1].add(word)
                    sentences[-1].append(str(word))
                else:
                    sentences[-1].append(word)

        return [
            (list(entities[i]), sentences[i])
            for i in range(len(sentences))
            if len(sentences[i]) > 0
        ]

    def get_negatives(self, nr_negatives, gender):
        if nr_negatives < 1:
            return []
        possible_corruptions = [
            x for x in (labels if gender else labels3) if x != self.answer
        ]
        negative_labels = choices(possible_corruptions, k=nr_negatives)
        negatives = []
        for label in negative_labels:
            copy = Story(copy=self)
            copy.answer = label
            copy.positive = False
            negatives.append(copy)
        return negatives

    def remove_gender(self):
        self.answer = remove_gender(self.answer)
        self.edges = {k: remove_gender(self.edges[k]) for k in self.edges}

    def to_query(self, query_type):
        if query_type == "split":
            return self.to_query_split()
        elif query_type == "edges":
            return self.to_query_edges()
        else:
            return self.to_query_whole()

    def get_text(self):
        return " ".join(str(x) for x in self.text)

    def get_relations(self):
        return self.edges

    def to_query_edges(self):
        entities = [Constant(x) for x in sorted(self.entities.values())]
        x, y = (
            Constant(self.entities[self.query[0]]),
            Constant(self.entities[self.query[1]]),
        )

        edges = [
            Term(self.edges[e], Constant(e[0]), Constant(e[1])) for e in self.edges
        ]

        query_term = Term(
            "clutrr_edges",
            list2term(entities),
            list2term(edges),
            x,
            Term(self.answer),
            y,
        )
        return Query(query_term, output_ind=[-2], p=float(self.positive))

    def to_query_whole(self):
        entities = [Constant(x) for x in sorted(self.entities.values())]
        substitution = dict()
        x, y = (
            Constant(self.entities[self.query[0]]),
            Constant(self.entities[self.query[1]]),
        )
        substitution[Term("text")] = Constant(
            '"' + " ".join(str(w) for w in self.text) + '"'
        )
        query_term = Term(
            "query_rel",
            Term("s", list2term(entities), Term("text")),
            x,
            Term(self.answer),
            y,
        )
        return Query(query_term, substitution, output_ind=[-2], p=float(self.positive))

    def to_query_split(self):
        entities = [Constant(x) for x in sorted(self.entities.values())]
        x, y = (
            Constant(self.entities[self.query[0]]),
            Constant(self.entities[self.query[1]]),
        )
        sentences = []
        for e, s in self.get_sentences():
            e = list2term([Constant(x) for x in e])
            s = Constant(" ".join(s))
            sentences.append(Term("s", e, s))
        query_term = Term(
            "clutrr_text",
            list2term(entities),
            list2term(sentences),
            x,
            Term(self.answer),
            y,
        )
        return Query(query_term, output_ind=[-2], p=float(self.positive))


class CLUTRR_Dataset(Dataset):
    def __init__(self, name, stories, vocab, gender, query_type, nr_negatives=0):
        self.gender = gender
        self.name = name
        self.type = query_type
        self.vocab = vocab
        negatives = []
        for story in stories:
            negatives += story.get_negatives(nr_negatives, gender)
        self.stories = stories + negatives

    def __getitem__(self, item):
        if not self.gender:
            self.stories[item].remove_gender()
        return self.stories[item]

    def to_query(self, i, j=None):
        if not self.gender:
            self.stories[i].remove_gender()
        return self.stories[i].to_query(self.type)

    def __len__(self):
        return len(self.stories)

    def __repr__(self):
        return dataset_names[self.name]


class CLUTRR(object):
    def __init__(self, name):
        self.synonym = name
        name = synonyms.get(name, name)
        self.name = Path(name)
        self.vocab = defaultdict(int)
        self.data = dict()
        self.subsets = self.name.glob("*.csv")
        for subset in self.subsets:
            self.load_file(subset)
        self.vocab = (
                ["ENT{}".format(x) for x in range(10)]
                + ["ENT", "OOV"]
                + sorted(self.vocab, key=lambda x: self.vocab[x], reverse=True)
        )

    def load_file(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            self.data[filename.stem] = [Story(line, self.vocab) for line in reader]

    def get_vocabulary(self):
        return {w: i for i, w in enumerate(self.vocab)}

    def save_vocabulary(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.vocab))

    def load_vocabulary(self, path):
        with open(path) as f:
            self.vocab = [word.rstrip("\n") for word in f]

    def get_embeddings(self, path="glove.6B.50d.txt", n=50):
        weights = torch.empty((len(self.vocab), n))
        weights.normal_(0, 1)
        vocab = self.get_vocabulary()
        with open(path) as f:
            for line in f:
                line = line.strip().split(" ")
                word, vec = line[0], [float(x) for x in line[1:]]
                try:
                    i = vocab[word]
                    weights[i, :] = torch.tensor(vec)
                except KeyError:
                    pass
        return weights

    def get_dataset(self, pattern, separate=False, **kwargs) -> Union[dict[CLUTRR_Dataset], CLUTRR_Dataset]:
        if separate:
            datasets = dict()
            for k in self.data:
                if match(pattern, k):
                    datasets[k] = CLUTRR_Dataset(
                        self.synonym, self.data[k], self.get_vocabulary(), **kwargs
                    )
            return datasets
        else:
            stories = []
            for k in self.data:
                if match(pattern, k):
                    stories += self.data[k]
            return CLUTRR_Dataset(
                self.synonym, stories, self.get_vocabulary(), **kwargs
            )

    def get_relations(self, pattern, gender):
        data = []
        for k in self.data:
            if match(pattern, k):
                for story in self.data[k]:
                    if gender is False:
                        story.remove_gender()
                    entities = list(story.entities.values())
                    data.append((story.get_text(), entities, story.get_relations()))
        return data
