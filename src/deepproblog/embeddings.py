from abc import ABC, abstractmethod
from typing import List, Optional, Collection

import torch

from problog.logic import Term


class TermEmbedder(ABC):
    """
    Generic embedder for terms.
    """

    @abstractmethod
    def get_embedding(self, term: Term) -> torch.Tensor:
        """

        :param term: Term to be embedded
        :return:
        """
        pass

    def zero_grad(self):
        """
        Zeroes the gradients for the learnable tensors.
        """
        pass

    def step(self):
        """
        Perform calls the step method for the optimizers of the contained learned embeddings.
        """
        pass


class OneHotEncoding(TermEmbedder):
    """
    One-hot embedder. Assign each unique term a unique one-hot embedding.
    """

    def __init__(self, n: int):
        """

        :param n: The dimension of the embeddings.
        """
        self.n = n
        self.embedding_indices = dict()
        self.nr_embeddings = 0

    def _add_embedding(self, term):
        i = self.nr_embeddings
        self.nr_embeddings += 1
        if self.nr_embeddings > self.n:
            raise Exception("Number of one-hot encodings exceeds dimensions.")
        self.embedding_indices[term] = i
        return i

    def get_embedding(self, term):
        term = str(term)
        try:
            i = self.embedding_indices[term]
        except KeyError:
            i = self._add_embedding(term)
        tensor = torch.zeros((1, self.n))
        tensor[0, i] = 1.0
        return tensor


class Embedding(TermEmbedder):
    """
    Embeds terms with learnable embeddings. The memory for the embeddings is allocated in blocks.
    """

    def __init__(self, n: int, block_size: int = 32, renormalize: bool = False):
        """

        :param n: The dimension of the embeddings.
        :param block_size: How many embeddings are allocated per block.
        :param renormalize: If true, then each embedding is normalized after the update step.
        """
        self.embedding_indices = dict()
        self.keys = []
        self.optimizer = None
        self.renormalize = renormalize
        self.n = n
        self.block_size = block_size
        self.embeddings = [self._get_new_embeddings()]
        self.nr_embeddings = 0

    def _get_new_embeddings(self):
        return torch.nn.Embedding(self.block_size, self.n)

    def _grow(self):
        self.embeddings.append(self._get_new_embeddings())
        self.optimizer.add_param_group({"params": self.embeddings[-1].parameters()})

    def _add_embedding(self, term):
        i = self.nr_embeddings
        self.nr_embeddings += 1

        if i // self.block_size >= len(self.embeddings):
            self._grow()
        self.embedding_indices[term] = i
        self.keys.append(term)
        return i

    def get_embedding(self, term):
        term = str(term)
        try:
            i = self.embedding_indices[term]

        except KeyError:
            i = self._add_embedding(term)
        return self.embeddings[i // self.block_size](
            torch.LongTensor([i % self.block_size])
        )

    def parameters(self) -> List[torch.nn.Parameter]:
        """

        :return: A list of all parameters of the embeddings.
        """
        return [
            param for embedding in self.embeddings for param in embedding.parameters()
        ]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        if self.renormalize:
            for embed in self.embeddings:
                for j in range(embed.weight.shape[0]):
                    embed.weight[j, :].data /= 0.1 * embed.weight[j, :].detach().norm()

    def get_all_embeddings(self, n: Optional[int] = None) -> torch.Tensor:
        """

        :param n: Optional int. Limits the returned embeddings to the first n.
        :return: A tensor containing the embeddings.
        """
        if n is None:
            n = self.nr_embeddings
        return torch.cat([embedding.weight for embedding in self.embeddings])[:n]

    def get_embeddings_by_label(self, labels: Collection[Term]) -> torch.Tensor:
        """
        :param labels: A list of terms.
        :return: Returns the tensor of the embeddings of the terms in label in order.
        """
        return torch.cat([self.get_embedding(label) for label in labels])
