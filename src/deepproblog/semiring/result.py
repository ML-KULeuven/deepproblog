from typing import Dict, Union, Optional, TYPE_CHECKING

import torch
from problog.formula import LogicFormula
from problog.logic import Term

if TYPE_CHECKING:
    from .semiring import Semiring


class Result(object):
    """
    A class that contains the result and timing info for evaluating a query.
    """

    def __init__(
        self,
        result: Dict[Term, Union[float, torch.Tensor]],
        semiring: 'Semiring',
        ground_time: Optional[float] = None,
        compile_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        proof: Optional[LogicFormula] = None,
    ):
        """Construct object

        :param result: Dictionary of results, the key is the term and the value is the probability.
        :param semiring: Semiring object in use
        :param ground_time:
        :param compile_time:
        :param eval_time:
        :param proof:

        Note! The term indexing the result object may not be the same as in your query. There are
        a few reasons for this:
        * Your query had substitutions, the term is going to be the substituted variant.
        * You have a non-ground query, your query could be partially ground (giving you multiple answers).
        """
        self.result = result
        self.semiring = semiring
        self.ground_time = ground_time
        self.compile_time = compile_time
        self.eval_time = eval_time
        self.proof = proof

    def prediction(self):
        keys = list(self.result.keys())
        if len(keys) != 1:
            raise ValueError('Cannot get prediction for a result with {} answers.'.format(len(self.result)))
        result = self.result[keys[0]]
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, requires_grad=True)
        return result

    def __iter__(self):
        return iter(self.result.keys())

    def __getitem__(self, item):
        return self.result[item]

    def __len__(self):
        return len(self.result)

    def __repr__(self):
        return repr(self.result)
