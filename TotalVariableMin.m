clc;clear all;close all;
rand('seed', 0);
randn('seed', 0);
n = 100;
x0 = ones(n,1);

for j = 1:3
    idx = randsample(n,1);
    k = randsample(1:10,1);
    x0(ceil(idx/2):idx) = k*x0(ceil(idx/2):idx);
end

b = x0 + randn(n,1);
lambda = 5;
e = ones(n,1);
D = spdiags([e -e], 0:1, n,n);

x = total_variation(b, lambda, 1.0, 1.0);

function x = total_variation(b, lambda, rho, alpha)
% total_variation  Solve total variation minimization via ADMM
% Solves the following problem via ADMM:
%
%   minimize  (1/2)||x - b||_2^2 + lambda * sum_i |x_{i+1} - x_i|
% where b in R^n.
%
% The solution is returned in the vector x.
%
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter
    MAX_ITER = 1000;
    n = length(b);

    % difference matrix
    e = ones(n,1);
    D = spdiags([e -e], 0:1, n,n);
    x = zeros(n,1);
    z = zeros(n,1);
    u = zeros(n,1);
    I = speye(n);
    DtD = D'*D;

    for k = 1:MAX_ITER

        % x-update
        x = (I + rho*DtD) \ (b + rho*D'*(z-u));

        % z-update with relaxation
        zold = z;
        Ax_hat = alpha*D*x +(1-alpha)*zold;
        z = shrinkage(Ax_hat + u, lambda/rho);

        % y-update
        u = u + Ax_hat - z;
    end
end    

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end
