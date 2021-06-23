nn(net1, [X], Y, [heads, tails]) :: coin(1,X,Y).
nn(net2, [X], Y, [heads, tails]) :: coin(2,X,Y).

outcome(X,X,win).
outcome(X,Y,loss) :- \+outcome(X,Y,win).

game(X,Outcome) :-
    coin(1,X,C1),
    coin(2,X,C2),
    outcome(C1,C2,Outcome).

and([]).
and([H|T]) :- H, and(T).