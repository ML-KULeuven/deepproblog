%get_ucs_value(_,_,0) :- !.

get_ucs_value(Key,Value,P) :-
    flag(count(Key),N_key,N_key+1),
    flag(count(Key,Value),N_value,N_value+1),
    writeln(count(N_key, N_value)),
    ((N_key == 0 ; N_value == 0) ->
        P = 1
    ;
        P is min(1,sqrt(2*log(N_key)/N_value))
    ).