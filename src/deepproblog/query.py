from typing import Optional, List, Sequence, Dict

from problog.logic import Term, Var


class Query(object):
    """
    The query object.
    """

    def __init__(
        self,
        query: Term,
        substitution: Optional[Dict[Term, Term]] = None,
        p: float = 1.0,
        output_ind: Sequence[int] = (-1,),
        collect=False,
    ):
        """

        :param query: The query term.
        :param substitution: An optional substitution that can be applied to the term (see Term.apply_term).
        Used for parameterized queries.
        :param p: The target probability of this query.
        :param output_ind: A tuple that contains the indices of the arguments that should be considered output
        arguments. This is relevant for testing / negative mining.
        """
        self.query = query
        self.substitution: Dict[Term, Term] = substitution
        if self.substitution is None:
            self.substitution = {}
        self.p = p
        self.output_ind = output_ind
        self.collect = collect

    def variable_output(self) -> "Query":
        """

        :return:  A new query identical to the current one, but with the output arguments replaced with variables.
        """
        new_args = list(self.query.args)
        for i in self.output_ind:
            new_args[i] = Var("X_{}".format(i % len(new_args)))
        return Query(self.query(*new_args), self.substitution, self.p, self.output_ind)

    def replace_output(self, new_values: List[Term]) -> "Query":
        """
        Replaces the output variables
        :param new_values: The new values in order that should replace the output variables.
        :return: The query with the out_variables replaced by the corresponding new values.
        """
        new_args = list(self.query.args)
        j = 0
        for i in self.output_ind:
            new_args[i] = new_values[j]
            j += 1
        return Query(self.query(*new_args), self.substitution, self.p, self.output_ind)

    def output_values(self, term: Optional[Term] = None) -> List[Term]:
        """
        :param term: Optional parameter. Will get the output value of the given term instead of 'self.query'
        :return: The values of the output arguments
        """
        if term is None:
            term = self.query
        return [term.args[i] for i in self.output_ind]

    def output_value(self, term: Optional[Term] = None) -> Term:
        """
        :param term: Optional parameter. Will get the output value of the given term instead of 'self.query'
        :return: The value of the output arguments
        """
        if term is None:
            term = self.query
        if len(self.output_ind) != 1:
            raise ValueError(
                "Cannot call output_value on a query with multiple output indices."
            )
        return term.args[self.output_ind[0]]

    def substitute(self, substitution: Optional[Dict[Term, Term]] = None) -> "Query":
        """

        :param substitution: The dictionary that will be used to perform the substitution.
                             If None, then 'self.substitution' will be used instead.
        :return: A new query where the substitution is applied. See the apply_term
                 method from the Term class for details on the substitution.
        """
        if substitution is None:
            substitution = self.substitution
        return Query(
            self.query.apply_term(substitution),
            substitution=None,
            p=self.p,
            output_ind=self.output_ind,
        )

    def change_substitution(self, new_substitution: dict) -> "Query":
        new_substitution2 = {
            new_substitution[k]: self.substitution[k] for k in new_substitution
        }
        return Query(
            self.query.apply_term(new_substitution),
            substitution=new_substitution2,
            p=self.p,
            output_ind=self.output_ind,
        )

    def __repr__(self):
        return "({}::{}, {})".format(self.p, self.query, self.substitution)

    def __hash__(self):
        return hash(self.query)

    def __eq__(self, other):
        if not isinstance(other, Query):
            return False
        return (
            self.query == other.query
            and self.substitution == other.substitution
            and self.p == other.p
            and self.output_ind == other.output_ind
        )

    def _get_tensor_substitution(self, term: Term, substitution: dict):
        if term.functor == "tensor":
            if term not in substitution:
                key = Term("arg_{}".format(len(substitution)))
                substitution[term] = key
        else:
            for arg in term.args:
                self._get_tensor_substitution(arg, substitution)

    def substitute_tensors(self) -> "Query":
        substitution = dict()
        self._get_tensor_substitution(self.query, substitution)
        new_term = self.query.apply_term(substitution)
        substitution = {substitution[key]: key for key in substitution}
        substitution.update(self.substitution)
        return Query(new_term, substitution, self.p, self.output_ind, self.collect)
