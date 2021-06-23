nn(swap_net, [X,Y],Z,[no_swap, swap]) :: swap(X,Y,Z).

hole(X,Y,X,Y):-
    swap(X,Y,no_swap).

hole(X,Y,Y,X):-
    swap(X,Y,swap).

bubble([X],[],X).
bubble([H1,H2|T],[X1|T1],X):-
    hole(H1,H2,X1,X2),
    bubble([X2|T],T1,X).

bubblesort([],L,L).

bubblesort(L,L3,Sorted) :-
    bubble(L,L2,X),
    bubblesort(L2,[X|L3],Sorted).

forth_sort(L,L2) :- bubblesort(L,[],L2).