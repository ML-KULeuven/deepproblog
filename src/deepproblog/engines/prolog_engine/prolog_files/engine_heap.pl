:- dynamic fa/4, cl/3.
:- include("heuristics_heap.pl").
:- include("builtins.pl").

:- set_prolog_flag(stack_limit, 12 000 000 000).
:- table neural_pred/3, get_ucs_value/3.
:- set_flag(mode,train), set_flag(iter,0).



%Node format:
% Heuristic-n([GoalProof-Goal| GoalsTail],Proof,Context)

get_node_child(Heuristic-n([CallProof-CallNode|GoalsTail],Proof,Context),Heuristic2-n([CallProof-Node|GoalsTail],Proof,Context)) :-
    CallNode =.. [call,Goal|ExtraArgs],!,
    Goal =.. [Call|Args],
    append(Args,ExtraArgs,AllArgs),
    Node =.. [Call|AllArgs],
    goal_heuristic([CallProof-Node|GoalsTail],Heuristic, Heuristic2).

get_node_child(Heuristic-n([GoalProof-and(BodyList)|GoalsTail],Proof,Context), Heuristic2-n(NewGoals,Proof,Context)) :-
    pairs_keys_values(BodyGoals,BodyGoalProofs,BodyList),
    GoalProof = and(BodyList)-and(BodyGoalProofs),
    append(BodyGoals,GoalsTail,NewGoals),
    goal_heuristic(NewGoals,Heuristic,Heuristic2).


get_node_child(Heuristic-n([GoalProof-Builtin|GoalsTail],Proof,Context), Heuristic2-n(GoalsTail,Proof,Context)) :-
    allowed_builtin(Builtin),!,
    Builtin,
    GoalProof = Builtin-builtin,
    goal_heuristic(GoalsTail,Heuristic, Heuristic2).

get_node_child(Heuristic-n([GoalProof-extern(Extern)|GoalsTail],Proof,Context), Heuristic2-n(GoalsTail,Proof,Context)) :-
    !,
    Extern,
    GoalProof = Extern-extern,
    goal_heuristic(GoalsTail,Heuristic, Heuristic2).


get_node_child(Heuristic-n([GoalProof-Fact|GoalsTail],Proof,Context), NewHeuristic-n(GoalsTail,Proof,Context)) :-

    fa(Id,Label,Fact,Ad),
    GoalProof = -(Fact,::(Id,Label,Fact,Ad)),
    increase_depth(Heuristic, Heuristic2),
    goal_heuristic(GoalsTail,Heuristic2, Heuristic3),
    (Label = nn(Net,Input,Index) ->

        atomic_concat(Net,"_extern",FuncName),
        neural_pred(FuncName,Input,Probabilities),
        nth0(Index,Probabilities,Probability),
        (get_flag(mode,train),get_flag(exploration,true) ->
            get_ucs_value(Net,Index,UCS),
            Probability_UCS is Probability + UCS - Probability*UCS,
            add_probability_to_heuristic(Probability_UCS,Heuristic3,NewHeuristic)
        ;
            add_probability_to_heuristic(Probability,Heuristic3,NewHeuristic))
    ;
        (Label = nn(Net,Input) ->
        atomic_concat(Net,"_extern",FuncName),
        neural_pred(FuncName,Input,Probability),
        %Todo: no exploration for neural facts yet
        add_probability_to_heuristic(Probability,Heuristic3,NewHeuristic)
        ;
            (Label = t(Parameter) ->
                get_parameter(Parameter,P),
                add_probability_to_heuristic(P,Heuristic3,NewHeuristic)
            ;
                (Label = tensor(T) ->
                    get_tensor_probability(T,P),
                    add_probability_to_heuristic(P,Heuristic3,NewHeuristic)
                ;
                    add_probability_to_heuristic(Label,Heuristic3,NewHeuristic)
                )
            )
        )
    ).




get_node_child(Heuristic-n([GoalProof-Clause|GoalsTail],Proof,Context), Heuristic3-n(NewGoals,Proof,Context)) :-
    cl(_,Clause,BodyList),
    increase_depth(Heuristic, Heuristic2),
    pairs_keys_values(BodyGoals,BodyGoalProofs,BodyList),
    GoalProof = Clause-and(BodyGoalProofs),
    append(BodyGoals,GoalsTail,NewGoals),
    goal_heuristic(NewGoals,Heuristic2,Heuristic3).



step(Heap,_,Proofs,Proofs,_) :- empty_heap(Heap),!.

step(_,K,Proofs,Proofs,_) :-  length(Proofs,Length),Length >= K, !.

step(_,_,Proofs,Proofs,0) :- !.

step(Heap,K,ProofAccumulator,Proofs,Depth) :-
    NextDepth is Depth - 1,
    get_from_heap(Heap,Heuristic,Node,NextHeap),!,
    Selected = Heuristic-Node,
    Node = n(Goals,_,_),
    maplist(record_selected_stored_heuristic(Heuristic),Goals),
    (isproven(Selected) ->
        !, step(NextHeap,K,[Depth-Selected|ProofAccumulator],Proofs,NextDepth)
    ;
        (get_all_node_children(Selected,Children) ->
            list_to_heap(Children,ChildHeap),
            merge_heaps(NextHeap,ChildHeap,TotalHeap),
            !,step(TotalHeap,K,ProofAccumulator,Proofs,NextDepth)
        ;
            !, step(NextHeap,K,ProofAccumulator,Proofs,NextDepth)
        )
    ).

prove(Query,K,FinalProofs,Heuristic,Exploration) :-
    set_flag(exploration,Exploration),
    (get_flag(mode,train) -> flag(iter,Iter,Iter+1);true),
    erase_recorded_heuristics,
    abolish_all_tables,
    call(Heuristic,H),
    singleton_heap(Heap,H,n([Proof-Query],Proof,[])),
    step(Heap,K,[],Proofs,-1),!,
    maplist(finalize(-1),Proofs,FinalProofs).

finalize(Depth,D-(_-n(_,X,_)),D2-X) :- D2 is Depth-D.

get_ucs_value(Key,Value,P) :-
    flag(Key,N_key,N_key+1),
    flag(Value,N_value,N_value+1),
    ((N_key == 0 ; N_value == 0) ->
        P = 1
    ;
        P is min(1,sqrt(2*log(N_key)/N_value))
    ).

neural_pred(FuncName,Input,Probs) :- call(FuncName,Input,Probs).

get_all_node_children(Node,[ChildHead|ChildTail]) :- findall(X, get_node_child(Node,X), [ChildHead|ChildTail]).

isproven(_-n([],_,_)).