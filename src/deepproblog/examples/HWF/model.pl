nn(net1,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: detect_number(X,Y).
nn(net2,[X],Y,[+,-,*,/]) :: detect_operator(X,Y).

detect_all([N],[N2]) :- detect_number(N,N2).
detect_all([N,O|T],[N2,O2|T2]) :- detect_number(N,N2), detect_operator(O,O2), detect_all(T,T2).

almost_equal(X,Y) :- ground(Y),abs(X-Y) < 0.0001.
almost_equal(X,Y) :- var(Y), Y is float(X).

expression(Images,Result) :- detect_all(Images,Symbols),parse(Symbols,Result).

parse([N],R) :-almost_equal(N,R).

parse([N1,+|T], R) :-
    parse(T,R2),
    almost_equal(N1+R2,R).

parse([N1,-|T], R) :-
    parse([-1,*|T],R2),
    almost_equal(N1+R2,R).

parse([N1,*,N2|T], R) :-
    N3 is N1*N2,
    parse([N3|T],R).

parse([N1,/,N2|T], R) :-
    N2 \== 0,
    N3 is N1/N2,
    parse([N3|T],R).