clc;clear all;close all;
m = 5000;       % number of examples
n = 200;        % number of features

x0 = randn(n,1);
A = randn(m,n);
A = A*spdiags(1./norms(A)',0,n,n); % normalize columns
b = A*x0 + sqrt(0.01)*randn(m,1);
b = b + 10*sprand(m,1,200/m);      % add sparse, large noise
[x,z,u] = huber_fit(A, b, 1.0, 1.0);

function [x,z,u] = huber_fit(A, b, rho, alpha)
% huber_fit  Solves a robust fitting problem
% solves the following problem via ADMM:
%
%   minimize 1/2*sum(huber(A*x - b))
%
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter

    MAX_ITER = 1000;
    [m, n] = size(A);
    
    x = zeros(n,1);
    z = zeros(m,1);
    u = zeros(m,1);

    % cache factorization
    [L U] = factor(A);
    for k = 1:MAX_ITER

        % x-update
        x = inv(A'*A)*A'*(b+z-u);

        % z-update with relaxation
        zold = z;
        Ax_hat = alpha*A*x + (1-alpha)*(zold + b);
        tmp = Ax_hat - b + u;
        z = rho/(1 + rho)*tmp + 1/(1 + rho)*shrinkage(tmp, 1 + 1/rho);

        u = u + (Ax_hat - z - b);
    end
end

function [L U] = factor(A)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A, 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end

function z = shrinkage(x, kappa)
    z = pos(1 - kappa./abs(x)).*x;
end