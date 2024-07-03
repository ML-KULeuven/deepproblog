from typing import TYPE_CHECKING

from problog.engine import DefaultEngine
from problog.extern import problog_export

from .engine import Engine
from .builtins import register_tensor_predicates

if TYPE_CHECKING:
    from ..model import Model


class ExactEngine(Engine):
    def __init__(self, model: "Model", cache_memory: bool = False, cache_root: str = None):
        Engine.__init__(self, model, cache_memory, cache_root)
        self.engine = DefaultEngine()
        self.program = self.engine.prepare(self.model.program)
        register_tensor_predicates(self)

    def ground(self, query):
        ground = self.engine.ground(self.program, query.query, label='query')
        return ground

    def register_foreign(self, func, function_name, arity_in, arity_out):
        signature = ["+term"] * arity_in + ["-term"] * arity_out
        problog_export.database = self.program
        problog_export(*signature)(func, funcname=function_name, modname=None)

