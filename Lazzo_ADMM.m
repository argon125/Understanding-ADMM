clc;clear all;close all;
m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
b = A*x0 + sqrt(0.001)*randn(m,1);

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;
[x,z,u] = lasso(A, b, lambda, 1.0, 1.0);
function [x,z,u] = lasso(A, b, lambda, rho, alpha)
% lasso  Solve lasso problem via ADMM
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
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
        x_hat = alpha*x + (1 - alpha)*zold;
        z = shrinkage(x_hat + u, lambda/rho);

        % u-update
        u = u + (x_hat - z);
    end
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
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