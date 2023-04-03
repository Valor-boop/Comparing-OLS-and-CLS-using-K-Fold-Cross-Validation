syms w1 w2 g1 g2 lambda
g = [g1;g2];
f = 0.5*(transpose([w1;w2]-g)*([w1;w2]-g));
% contraint equation
Pa = w1^2 + w2^2 - 1;
% jacob matrix
Jpa = jacobian(Pa)
g1 = gradient(f,[w1,w2])

% compute minimizer using stationarity 
% solving equation 3 to get lambda 
% set w2 to 2 for 
stat = transpose(subs(g1,w1,1) + lambda*subs(Jpa, w1, w2))
% solve lagrand when w = 0 
[multiplier w1a] = solve(stat)

%lagrand and gradient 
L1 = f + multiplier*Pa
L1g = gradient(L1, [w1,w2])

% solve for 0 of lagrange equation
% where legrange gradient is equal to 0
[min1a, min2a] = solve(L1g)
g1 = @(g1, g2)((g1*(g2 - g1 + 1))/g2)/g1
g2 = @(g1, g2) (g2 - g1 + 1)/g2


g12 = g1(1.5, 0)
g22 = g2(1.5, 0)
[1;1] + ([-0.7;-0.7])*(sqrt(2)-1)
sind(45)
norm(g22)
