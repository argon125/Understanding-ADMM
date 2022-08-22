clc;clear all;close all;
randn('seed', 0);
rand('seed',0);

m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

% generate sparse solution vector
x = sprandn(n,1,p);

% generate random data matrix
A = randn(m,n);

% normalize columns of A
A = A*spdiags(1./sqrt(sum(A.^2))', 0, n, n);

% generate measurement b with noise
b = A*x + sqrt(0.001)*randn(m,1);

xtrue = x;   % save solution
x  = regressor_sel(A, b, p*n, 1.0);
function z = regressor_sel(A, b, K, rho)
% regressor_sel  Solve lasso problem via ADMM
%
% Attempts to solve the following problem via ADMM:
%
%   minimize || Ax - b ||_2^2
%   subject to card(x) <= K
%
% where card() is the number of nonzero entries.
%
% The solution is returned in the vector x.
%
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter 
    MAX_ITER = 1000;
    [m, n] = size(A);

    % save a matrix-vector multiply
    Atb = A'*b;
    x = zeros(n,1);
    z = zeros(n,1);
    u = zeros(n,1);

    % cache the factorization
    [L U] = factor(A, rho);
    for k = 1:MAX_ITER

        % x-update
        q = Atb + rho*(z - u);    % temporary value
        if( m >= n )    % if skinny
           x = U \ (L \ q);
        else            % if fat
           x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
        end

        % z-update with relaxation
        zold = z;
        z = keep_largest(x + u, K);

        % u-update
        u = u + (x - z);
    end
end

function z = keep_largest(z, K)
    [val,pos] = sort(abs(z), 'descend');
    z(pos(K+1:end)) = 0;
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end