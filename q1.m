syms w1 w2 lambda
g = [3;3];
f = 0.5*(transpose([w1;w2]-g)*([w1;w2]-g));
% contraint equation
Pa = w1^2 - 4;
% jacob matrix
Jpa = jacobian(Pa)
g1 = gradient(f,[w1,w2])

% compute minimizer using stationarity 
% solving equation 3 to get lambda 
% set w2 to 2 for 
stat = transpose(subs(g1,w1,2) + lambda*subs(Jpa, w1, 2))
% solve lagrand when w = 0 
[multiplier w1a] = solve(stat)

%lagrand and gradient 
L1 = f + 0.25*Pa
L1g = gradient(L1, [w1,w2])

% solve for 0 of lagrange equation
% where legrange gradient is equal to 0
[min1a, min2a] = solve(L1g)
