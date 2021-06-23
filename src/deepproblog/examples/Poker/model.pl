:-use_module(library(lists)).

t(1/4)::house_rank(jack);t(1/4)::house_rank(queen);t(1/4)::house_rank(king);t(1/4)::house_rank(ace).

nn(net1,[X],Z,[jack,queen,king,ace]):: rank(X,Z).


hand(Cards,straight(low)) :-
    member(card(jack),Cards),
    member(card(queen),Cards),
    member(card(king),Cards).

hand(Cards,straight(high)) :-
    member(card(queen),Cards),
    member(card(king),Cards),
    member(card(ace),Cards).

hand([card(R), card(R), card(R)],threeofakind(R)).


hand(Cards,pair(R)) :-
    select(card(R),Cards,Cards2),
    member(card(R),Cards2).

hand(Cards,high(R)) :-
    member(card(R),Cards).

hand_rank(high(jack),0).
hand_rank(high(queen),1).
hand_rank(high(king),2).
hand_rank(high(ace),3).
hand_rank(pair(jack),4).
hand_rank(pair(queen),5).
hand_rank(pair(king),6).
hand_rank(pair(ace),7).
hand_rank(threeofakind(jack),8).
hand_rank(threeofakind(queen),9).
hand_rank(threeofakind(king),10).
hand_rank(threeofakind(ace),11).
hand_rank(straight(low),12).
hand_rank(straight(high),13).

best_hand(Cards,H) :-
    hand(Cards,H),
    hand_rank(H,R),
    \+(hand(Cards,H2),hand_rank(H2,R2),R2>R).

best_hand_rank(Cards,R) :-
    hand(Cards,H),
    hand_rank(H,R),
    \+(hand(Cards,H2),hand_rank(H2,R2),R2>R).

outcome(R1,R2,win) :- R1 > R2.
outcome(R1,R2,loss) :- R1 < R2.
outcome(R,R,draw).

cards([C0,C1,_,_],own,House,[card(R1), card(R2), House]) :-
    rank(C0,R1),
    rank(C1,R2).

cards([_,_,C2,C3],opponent,House,[card(R1), card(R2), House]) :-
    rank(C2,R1),
    rank(C3,R2).

game(Cards,House,H1,H2) :-
    cards(Cards,own,House,C1),
    cards(Cards,opponent,House,C2),
    best_hand(C1,H1),
    best_hand(C2,H2).

game(Cards,House,Outcome) :-
    cards(Cards,own,House,C1),
    cards(Cards,opponent,House,C2),
    best_hand_rank(C1,R1),
    best_hand_rank(C2,R2),
    outcome(R1,R2,Outcome).

game(Cards,Outcome) :-
    house_rank(Rank),
    game(Cards,card(Rank),Outcome).

