nn(neural1,[I1,I2,Carry],O,[0,1,2,3,4,5,6,7,8,9]) :: result(I1,I2,Carry,O).
nn(neural2,[I1,I2,Carry],NewCarry,[0,1]) :: carry(I1,I2,Carry,NewCarry).

%nn(neural1,[I],O,[0,1,2,3,4,5,6,7,8,9]) :: result(I,O).
%nn(neural2,[I],NewCarry,[0,1]) :: carry(I,NewCarry).

slot(I1,I2,Carry,Carry2,O) :-
    result(I1,I2,Carry,O),
    carry(I1,I2,Carry,Carry2).
%    one_hot(I1,10,T1),
%    one_hot(I2,10,T2),
%    one_hot(Carry,2,T3),
%    cat([T1,T2,T3],T),
%    result(T,O),
%    carry(T,Carry2).

add([],[],C,C,[]).

add([H1|T1],[H2|T2],C,Carry,[Digit|Res]) :-
    add(T1,T2,C,Carry2,Res),
    slot(H1,H2,Carry2,Carry,Digit).

add(L1,L2,C,[Carry|Res]) :- add(L1,L2,C,Carry,Res).