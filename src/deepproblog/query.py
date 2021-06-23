from typing import Optional, List, Sequence

from problog.logic import Term, Var


class Query(object):
    """
    The query object.
    """

    def __init__(
        self,
        query: Term,
        substitution: Optional[dict] = None,
        p: float = 1.0,
        output_ind: Sequence[int] = (-1,),
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
        self.substitution = substitution
        if self.substitution is None:
            self.substitution = {}
        self.p = p
        self.output_ind = output_ind

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

    def output_values(self) -> List[Term]:
        """

        :return: The values of the output arguments
        """
        return [self.query.args[i] for i in self.output_ind]

    def substitute(self, substitution: Optional[dict] = None) -> "Query":
        """

        :param substitution: The dictionary that will be used to perform the substitution.
                             If None, then self.substitution will be used instead.
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

    def __repr__(self):
        return "({}::{}, {})".format(self.p, self.query, self.substitution)

    def to_text_query(self):
        """Return the query on a form where QueryDataset can load it"""
        subst = sorted(
            ([k, v] for k, v in self.substitution.items()), key=lambda x: x[0].functor
        )
        return f"{self.p}::subs({self.query}, {subst})."

    def __hash__(self):
        return (
            hash(self.query)
            ^ hash(self.substitution)
            ^ hash(self.p)
            ^ hash(self.output_ind)
        )

    def __eq__(self, other):
        if not isinstance(other, Query):
            return False
        return (
            self.query == other.query
            and self.substitution == other.substitution
            and self.p == other.p
            and self.output_ind == other.output_ind
        )
