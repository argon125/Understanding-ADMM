clc;clear all;close all;
n = 30;
m = 10;
A = randn(m,n);

x = sprandn(n, 1, 0.1*n);
b = A*x;

xtrue = x;
[X,Z,U] = basis_pursuit(A, b, 1.0, 1.0);

function [x,z,u] = basis_pursuit(A, b, rho, alpha)
% basis_pursuit  Solve basis pursuit via ADMM
%
% Solves the following problem via ADMM:
%   minimize     ||x||_1
%   subject to   Ax = b
%
% The solution is returned in the vector x.
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter
    MAX_ITER = 1000;
    [m n] = size(A);
    x = zeros(n,1);
    z = zeros(n,1);
    u = zeros(n,1);

    % precompute static variables for x-update (projection on to Ax=b)
    AAt = A*A';
    P = eye(n) - A' * (AAt \ A);
    q = A' * (AAt \ b);

    for k = 1:MAX_ITER
        % x-update
        x = P*(z - u) + q;

        % z-update with relaxation
        zold = z;
        x_hat = alpha*x + (1 - alpha)*zold;
        z = shrinkage(x_hat + u, 1/rho);

        u = u + (x_hat - z);
    end
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end
