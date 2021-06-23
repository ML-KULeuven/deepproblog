desc_keysort(X,Y) :- sort([1,1],@>=,X,Y).


merge_sorted_lists(X,[],X).
merge_sorted_lists([],X,X).

merge_sorted_lists([h(K1,T,L1)-H1|T1],[h(K2,T,L2)-H2|T2],[H3|T3]) :-
    (K1 > K2 ->
        H3 = h(K1,T,L1)-H1,!,
        merge_sorted_lists(T1,[h(K2,T,L2)-H2|T2], T3)
    ;
        H3 = h(K2,T,L2)-H2,!,
        merge_sorted_lists([h(K1,T,L1)-H1|T1],T2,T3)
    ).



limit_length(L,-1,L,none) :- !.

limit_length(L,Limit,L,none) :- length(L,Length), Length =< Limit, !.

limit_length(L,Limit,L2,X) :- length(L2,Limit), append([X|_],L2,L).


find_index([_],_,0) :- !.

find_index([H|T],Value,I) :-
    (Value =< H ->
        I = 0
    ;
        NextValue = Value-H,
        find_index(T,NextValue, I2),
        I is I2 + 1
    ).
