syms w1 w2 lambda
g = [1;1];
f = 0.5*(transpose([w1;w2]-g)*([w1;w2]-g));
% contraint equation
Pa = w2^2 - 4;
% jacob matrix
Jpa = jacobian(Pa)
g1 = gradient(f,[w1,w2])

% compute minimizer using stationarity 
stat = transpose(subs(g1,w1,2) + lambda*subs(Jpa, w1, w2))
% solve lagrand when w = 0

% solve for 0 of lagrange equation
[min1a, min2a] = solve(g1)