nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

t(0.1) :: noisy.

1/19::uniform(X,Y,0);1/19::uniform(X,Y,1);1/19::uniform(X,Y,2);1/19::uniform(X,Y,3);1/19::uniform(X,Y,4);1/19::uniform(X,Y,5);1/19::uniform(X,Y,6);1/19::uniform(X,Y,7);1/19::uniform(X,Y,8);1/19::uniform(X,Y,9);1/19::uniform(X,Y,10);1/19::uniform(X,Y,11);1/19::uniform(X,Y,12);1/19::uniform(X,Y,13);1/19::uniform(X,Y,14);1/19::uniform(X,Y,15);1/19::uniform(X,Y,16);1/19::uniform(X,Y,17);1/19::uniform(X,Y,18).

addition_noisy(X,Y,Z) :- noisy, uniform(X,Y,Z).
addition_noisy(X,Y,Z) :- \+noisy, digit(X,N1), digit(Y,N2), Z is N1+N2.

addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), Z is N1+N2.