P :: similar(X,Y) :- rbf(X,Y,P).

successor(X,Y,N) :- cnn_encode(X,EX), cnn_encode(Y,EY), embed(successor, S), mul(S,N,S2), add(EX,S2,EZ), similar(EZ,EY).