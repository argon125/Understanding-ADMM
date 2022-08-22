clc;clear all;close all;
n = 100;   % number of features
N = 10*n;  % number of samples

% generate a sparse positive definite inverse covariance matrix
Sinv      = diag(abs(ones(n,1)));
idx       = randsample(n^2, 0.001*n^2);
Sinv(idx) = ones(numel(idx), 1);
Sinv = Sinv + Sinv';   % make symmetric
if min(eig(Sinv)) < 0  % make positive definite
    Sinv = Sinv + 1.1*abs(min(eig(Sinv)))*eye(n);
end
S = inv(Sinv);

% generate Gaussian samples
D = mvnrnd(zeros(1,n), S, N);
[X] = covsel(D, 0.01, 1, 1);

function [Z] = covsel(D, lambda, rho, alpha)
% covsel  Sparse inverse covariance selection via ADMM
% Solves the following problem via ADMM:
%
%   minimize  trace(S*X) - log det X + lambda*||X||_1
%
% with variable X, where S is the empirical covariance of the data
% matrix D (training observations by features).
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter 

    MAX_ITER = 1000;

    S = cov(D);
    n = size(S,1);

    X = zeros(n);
    Z = zeros(n);
    U = zeros(n);

    for k = 1:MAX_ITER
        % x-update
        [Q,L] = eig(rho*(Z - U) - S);
        es = diag(L);
        xi = (es + sqrt(es.^2 + 4*rho))./(2*rho);
        X = Q*diag(xi)*Q';

        % z-update with relaxation
        Zold = Z;
        X_hat = alpha*X + (1 - alpha)*Zold;
        Z = shrinkage(X_hat + U, lambda/rho);

        U = U + (X_hat - Z);
    end
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end