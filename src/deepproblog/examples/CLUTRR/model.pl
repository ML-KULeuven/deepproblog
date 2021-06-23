
clutrr_text(Text,X,R,Y) :- get_edges(Text,Edges), clutrr_edges(Edges,X,R,Y).

clutrr_edges(Entities,Edges,X,R,Y) :- relation(R), length(Edges,D), call(R,[Entities|Edges],X,Y,D-1).
%clutrr_edges(Entities,Edges,X,R,Y) :- relation(R), length(Edges,LenEdges), D is ceiling(log(LenEdges)/log(2)), call(R,[Entities|Edges],X,Y,D).

edges_contain_relation([_|Edges],X,R,Y) :-  member(rel(X,R,Y),Edges).%, writeln(member(rel(X,R,Y),Edges)).

child(T,X,Y,_) :- edges_contain_relation(T,X,child,Y).
child_in_law(T,X,Y,_) :- edges_contain_relation(T,X,child_in_law,Y).
grandchild(T,X,Y,_) :-  edges_contain_relation(T,X,grandchild,Y).
parent(T,X,Y,_) :- edges_contain_relation(T,X,parent,Y).
parent_in_law(T,X,Y,_) :- edges_contain_relation(T,X,parent_in_law,Y).
grandparent(T,X,Y,_) :- edges_contain_relation(T,X,grandparent,Y).
sibling(T,X,Y,_) :- edges_contain_relation(T,X,sibling,Y).
sibling_in_law(T,X,Y,_) :- edges_contain_relation(T,X,sibling_in_law,Y).
so(T,X,Y,_) :- edges_contain_relation(T,X,so,Y).
uncle(T,X,Y,_) :- edges_contain_relation(T,X,uncle,Y).
nephew(T,X,Y,_) :- edges_contain_relation(T,X,nephew,Y).

select_entities([Entities|_],X,Y,Z) :- select(X,Entities,Ent2), select(Y, Ent2,Ent3), member(Z, Ent3).

grandchild(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), child(T,X,Z,D-1), child(T,Z,Y,D-1).
grandchild(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), so(T,X,Z,D-1), grandchild(T,Z,Y,D-1).
grandchild(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), grandchild(T,X,Z,D-1), sibling(T,Z,Y,D-1).
grandparent(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), parent(T,X,Z,D-1), parent(T,Z,Y,D-1).
grandparent(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), sibling(T,X,Z,D-1), grandparent(T,Z,Y,D-1).
child(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), child(T,X,Z,D-1), sibling(T,Z,Y,D-1).
child(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), so(T,X,Z,D-1), child(T,Z,Y,D-1).
parent(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), sibling(T,X,Z,D-1), parent(T,Z,Y,D-1).
parent(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), child(T,X,Z,D-1), grandparent(T,Z,Y,D-1).
sibling(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), child(T,X,Z,D-1), uncle(T,Z,Y,D-1).
sibling(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), parent(T,X,Z,D-1), child(T,Z,Y,D-1).
sibling(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), sibling(T,X,Z,D-1), sibling(T,Z,Y,D-1).
child_in_law(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), child(T,X,Z,D-1),so(T,Z,Y,D-1).
parent_in_law(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), so(T,X,Z,D-1), parent(T,Z,Y,D-1).
nephew(T,X,Y,D) :-D > 0, select_entities(T,X,Y,Z), sibling(T,X,Z,D-1), child(T,Z,Y,D-1).
uncle(T,X,Y,D) :- D > 0, select_entities(T,X,Y,Z), parent(T,X,Z,D-1), sibling(T,Z,Y,D-1).

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