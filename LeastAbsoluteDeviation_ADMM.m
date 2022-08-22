clc;clear all;close all;
rand('seed', 0);
randn('seed', 0);

m = 1000; % number of examples
n = 100;  % number of features

A = randn(m,n);
x0 = 10*randn(n,1);
b = A*x0;
idx = randsample(m,ceil(m/50));
b(idx) = b(idx) + 1e2*randn(size(idx));
[x,z,u] = lad(A, b, 1.0, 1.0);
function [x,z,u] = lad(A, b, rho, alpha)
% lad  Least absolute deviations fitting via ADMM
% Solves the following problem via ADMM:
%
%   minimize     ||Ax - b||_1
%
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter
    MAX_ITER = 1000;
    [m n] = size(A);
    x = zeros(n,1);
    z = zeros(m,1);
    u = zeros(m,1);

    for k = 1:MAX_ITER
        if k > 1
            x = R \ (R' \ (A'*(b + z - u)));
        else
            R = chol(A'*A);
            x = R \ (R' \ (A'*(b + z - u)));
        end

        zold = z;
        Ax_hat = alpha*A*x + (1-alpha)*(zold + b);
        z = shrinkage(Ax_hat - b + u, 1/rho);

        u = u + (Ax_hat - z - b);
    end
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end