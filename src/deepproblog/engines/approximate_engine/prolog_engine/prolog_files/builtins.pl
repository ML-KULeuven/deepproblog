:-dynamic allowed_builtin/1.

rstrip(_,[],[]) :- !.
rstrip(S,[S|T],[]) :-rstrip(S,T,[]),!.
rstrip(S,[H|T],[H|T2]) :- rstrip(S,T,T2).

allowed_builtin(writeln(_)).
allowed_builtin(write(_)).
allowed_builtin(writeq(_)).
allowed_builtin(true).
allowed_builtin(\==(_,_)).
allowed_builtin(==(_,_)).
allowed_builtin(\=(_,_)).
allowed_builtin(_>_).
allowed_builtin(_>=_).
allowed_builtin(_<_).
allowed_builtin(_=<_).
allowed_builtin(ground(_)).
allowed_builtin(var(_)).
allowed_builtin(is(_,_)).
allowed_builtin(maplist(_,_,_)).
allowed_builtin(embed(_,_)).
allowed_builtin(max(_,_)).
allowed_builtin(mean(_,_)).
allowed_builtin(rbf(_,_,_)).
allowed_builtin(add(_,_,_)).
allowed_builtin(mul(_,_,_)).
allowed_builtin(dot(_,_,_)).
allowed_builtin(sigmoid(_,_)).
allowed_builtin(member(_,_)).
allowed_builtin(length(_,_)).
allowed_builtin(select(_,_,_)).
allowed_builtin(nth0(_,_,_)).
allowed_builtin(transpose(_,_)).
allowed_builtin(include(_,_,_)).
allowed_builtin(convlist(_,_,_)).
allowed_builtin(rstrip(_,_,_)).
allowed_builtin(reverse(_,_)).
allowed_builtin(append(_,_,_)).
allowed_builtin(number(_)).
allowed_builtin(between(_,_,_)).
allowed_builtin(=..(_,_)).
allowed_builtin(=(_,_)).
allowed_builtin(\+(_)).
allowed_builtin(findall(_,_,_)).
allowed_builtin(forall(_,_)).
allowed_builtin(bagof(_,_,_)).