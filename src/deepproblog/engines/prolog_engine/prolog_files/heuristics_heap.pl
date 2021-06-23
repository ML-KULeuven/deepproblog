:- table get_heuristic/3.
:- dynamic stored_heuristic/2.
partial_probability(pp(-1.0)).
geometric_mean(gm(0.0,0)).
depth_first(dfs(0)).
breadth_first(bfs(0,-1.0)).
random(rnd(0)).
max_heuristic(Preds,max(-1,1,Preds)).
extern(Preds,extern(-1,1,Preds)). %First argument = (- cost * prediction), second argument = cost

add_probability_to_heuristic(Probability,gm(H1,D1),gm(H2,D2)) :- !, D2 is D1 + 1, H2 is (H1*D1-log(Probability))/D2.
add_probability_to_heuristic(Probability,pp(H1),pp(H2)) :- !, H2 is H1*Probability.
add_probability_to_heuristic(Probability,bfs(D,P1),bfs(D,P2)) :- !, P2 is P1*Probability.
add_probability_to_heuristic(Probability,max(C1,C2,Preds), max(C3,C4,Preds)) :- !,C3 is C1*Probability, C4 is C2*Probability.
add_probability_to_heuristic(Probability,extern(C1,C2,Preds), extern(C3,C4,Preds)) :- !,C3 is C1*Probability, C4 is C2*Probability.
add_probability_to_heuristic(_,X,X).


increase_depth(dfs(D1),dfs(D2)) :- !, D2 is D1 - 1.
increase_depth(bfs(D1,P),bfs(D2,P)) :- !, D2 is D1 + 1.
increase_depth(rnd(_),rnd(Random)) :- !, random(-1.0, 1.0, Random).
increase_depth(X,X) :- !.

multiply(X,Y,Z) :- Z is X*Y ,!.

update_heuristic(Key-Value) :-
    %writeln(k(K)),
    %K = Key-Value,
    %writeln(stored_heuristic(Key,_)),
    retractall(stored_heuristic(Key,_)),
    asserta(stored_heuristic(Key,Value)).
    %writeln(asserta(stored_heuristic(Key,Value))).

get_heuristic(Predicates,_-Goal, Heuristic) :-
    %writeln(Goal),
    ground(Goal),
    functor(Goal,Functor,_),
    member(Functor,Predicates),
    %writeln(goal=Goal),
    get_heuristic_extern(Goal, Heuristic),
   % writeln(get_heuristic_extern(Goal, Heuristic)),
    !.

get_heuristic(_,_,1).

goal_heuristic(Goals,max(_,Cost,Preds),max(Prediction,Cost,Preds)) :-
    !,
    maplist(get_heuristic_max(Preds),Goals,Heuristics),
    foldl(multiply,Heuristics,-1.0,GoalCost),
    Prediction is Cost*GoalCost.

goal_heuristic(Goals,extern(_,Cost,Predicates),extern(Prediction,Cost,Predicates)) :-
    !,
    maplist(get_heuristic(Predicates),Goals,Heuristics),
    foldl(multiply,Heuristics,-1.0,GoalCost),
    %writeln(gc=GoalCost),
    Prediction is Cost*GoalCost.

goal_heuristic(_,X,X) :- !.




get_heuristic_max(Preds,_-Goal, Heuristic) :-
    ground(Goal),
    functor(Goal,Functor,_),
    member(Functor,Preds),
    %(stored_heuristic(Goal,Heuristic),writeln(stored_heuristic(Goal,Heuristic)),!;Heuristic=1),
    (stored_heuristic(Goal,Heuristic),!;Heuristic=0),
    !.

get_heuristic_max(_,_,1).

record_selected_stored_heuristic(max(_,_,Preds),_-Goal) :-
    ground(Goal),
    functor(Goal,Functor,_),
    member(Functor,Preds),!,
    copy_term(Goal,GoalCopy),
    recordz(selected_heuristic,GoalCopy).

record_selected_stored_heuristic(_,_).

erase_recorded_heuristics :- forall(recorded(selected_heuristic,_,Ref),erase(Ref)).

get_recorded_heuristics(Recorded) :- findall(Record,recorded(selected_heuristic,Record,_),Recorded).

