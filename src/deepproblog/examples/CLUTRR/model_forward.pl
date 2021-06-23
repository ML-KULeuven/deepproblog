:-use_module(library(lists)).

nn(rel_extract, E, Relation,[child, child_in_law, parent, parent_in_law, sibling, sibling_in_law, grandparent, grandchild, nephew, uncle, so]) :: extract_relation(E,Relation).
nn(encoder, [Text,Ent1,Ent2], Embeddings) :: encode(Text,Ent1,Ent2,Embeddings).
nn(gender_net,[T,Entity],G,[male,female]) :: gender(T,Entity,G).



get_edge(s(Entities,Text), Edge) :-
    select(E1,Entities,Entities2),
    member(E2,Entities2),
    encode(Text,E1,E2,Encoded),
    extract_relation(Encoded,Relation),
    Edge =.. [Relation,E1,E2].


get_edges( [Sentence|Others], [Edge|Edges]) :-
    get_edge(Sentence,Edge),
    get_edges(Others,Edges).

get_edges([], []).


clutrr_text(Entities, Text,X,R2,Y) :-
    gender_rel(R,G,R2),
    gender(Text,Y,G),
    get_edges(Text,Edges),
    clutrr_edges(Entities,Edges,X,R,Y).

clutrr_edges(Entities,Edges,X,R,Y) :-
    relation(R),
    Goal =.. [R,X,Y],
    rules(Rules),
    forward(Edges,Rules, Goal).

forward(Facts,Rules, Goal) :-  member(Goal,Facts).
forward(Facts,Rules, Goal) :- \+member(Goal,Facts),forward_step(Facts,Rules,NextFacts),forward(NextFacts,Rules,Goal).

forward_step(Facts,Rules, AllFacts) :-
    findall(Head,(member(rule(Head, Body),Rules), maplist([X]>>member(X,Facts),Body), \+member(Head,Facts)), [New|NewFacts]),
    append([New|NewFacts],Facts,AllFacts).


rule(grandchild(X,Y),  [child(X,Z), child(Z,Y)]).
rule(grandchild(X,Y),  [so(X,Z), grandchild(Z,Y)]).
rule(grandchild(X,Y),  [grandchild(X,Z), sibling(Z,Y)]).
rule(grandparent(X,Y),  [parent(X,Z), parent(Z,Y)]).
rule(grandparent(X,Y),  [sibling(X,Z), grandparent(Z,Y)]).
rule(child(X,Y),  [child(X,Z), sibling(Z,Y)]).
rule(child(X,Y),  [so(X,Z), child(Z,Y)]).
rule(parent(X,Y),  [sibling(X,Z), parent(Z,Y)]).
rule(parent(X,Y),  [child(X,Z), grandparent(Z,Y)]).
rule(sibling(X,Y),  [child(X,Z), uncle(Z,Y)]).
rule(sibling(X,Y),  [parent(X,Z), child(Z,Y)]).
rule(sibling(X,Y),  [sibling(X,Z), sibling(Z,Y)]).
rule(child_in_law(X,Y),  [child(X,Z),so(Z,Y)]).
rule(parent_in_law(X,Y),  [so(X,Z), parent(Z,Y)]).
rule(nephew(X,Y), [sibling(X,Z), child(Z,Y)]).
rule(uncle(X,Y),  [parent(X,Z), sibling(Z,Y)]).
rules([rule(grandchild(X,Y),  [child(X,Z), child(Z,Y)]),  rule(grandchild(X,Y),  [so(X,Z), grandchild(Z,Y)]),  rule(grandchild(X,Y),  [grandchild(X,Z), sibling(Z,Y)]),  rule(grandparent(X,Y),  [parent(X,Z), parent(Z,Y)]),  rule(grandparent(X,Y),  [sibling(X,Z), grandparent(Z,Y)]),  rule(child(X,Y),  [child(X,Z), sibling(Z,Y)]),  rule(child(X,Y),  [so(X,Z), child(Z,Y)]),  rule(parent(X,Y),  [sibling(X,Z), parent(Z,Y)]),  rule(parent(X,Y),  [child(X,Z), grandparent(Z,Y)]),  rule(sibling(X,Y),  [child(X,Z), uncle(Z,Y)]),  rule(sibling(X,Y),  [parent(X,Z), child(Z,Y)]),  rule(sibling(X,Y),  [sibling(X,Z), sibling(Z,Y)]),  rule(child_in_law(X,Y),  [child(X,Z),so(Z,Y)]),  rule(parent_in_law(X,Y),  [so(X,Z), parent(Z,Y)]),  rule(nephew(X,Y), [sibling(X,Z), child(Z,Y)]),  rule(uncle(X,Y),  [parent(X,Z), sibling(Z,Y)])]).
relation(child).
relation(child_in_law).
relation(parent).
relation(parent_in_law).
relation(sibling).
relation(sibling_in_law).
relation(grandparent).
relation(grandchild).
relation(nephew).
relation(uncle).
relation(so).

gender_rel(child,male,son).
gender_rel(child,female,daughter).

gender_rel(parent,male,father).
gender_rel(parent,female,mother).

gender_rel(grandchild,male,grandson).
gender_rel(grandchild,female,granddaughter).

gender_rel(grandparent,male,grandfather).
gender_rel(grandparent,female,grandmother).

gender_rel(uncle,male,uncle).
gender_rel(uncle,female,aunt).

gender_rel(child_in_law,male,son_in_law).
gender_rel(child_in_law,female,daughter_in_law).

gender_rel(parent_in_law,male,father_in_law).
gender_rel(parent_in_law,female,mother_in_law).

gender_rel(nephew,male,nephew).
gender_rel(nephew,female,niece).

gender_rel(sibling,male,brother).
gender_rel(sibling,female,sister).

gender_rel(sibling_in_law,male,brother_in_law).
gender_rel(sibling_in_law,female,sister_in_law).

gender_rel(so,male,husband).
gender_rel(so,female,wife).