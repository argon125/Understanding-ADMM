clc;clear all;close all;
% Linear_Programming Solves the following problem via ADMM:
%
%   minimize     psi
%   subject to   0<= psi <= I + (Aw-gamma*e) 
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter

% n = 500;  % dimension of x
% m = 400;  % number of equality constraints
% x0 = abs(randn(n,1));    % create random solution vector
% 
% A = abs(randn(m,n));     % create random, nonnegative matrix A
% b = A*x0;

% N = 6;

% P = eye(n1);
% b = D*(A*w-gamma*e)+e;

randn('state', 0);
rand('state', 0);

% n = 500;  % dimension of x
% m = 400;  % number of equality constraints
% 
% c  = rand(n,1) + 0.5;    % create nonnegative price vector with mean 1
% x0 = abs(randn(n,1));    % create random solution vector
% 
% A = abs(randn(m,n));     % create random, nonnegative matrix A
% b = A*x0;
A = [1 2;2 1;2 2;3 3;3 4;4 3];
d = [-1;-1;-1;1;1;1]; D = diag(d); 
w = randi([0 9],2,1);
n = size(A);n1 = n(2); n2 = n(1);
gamma = 5; e = ones(n2,1);
b = D*(A*w-gamma*e)+e;

[z,hist] = linprog(A, b, 1.0, 1.0);%function call

function [z, history] = linprog( A, b, rho, alpha)
    QUIET    = 0;
    MAX_ITER = 1000;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    
    [m n] = size(A);
    x = zeros(n,1);
    z = zeros(n,1);
    u = zeros(n,1);

    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
          'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end

    for k = 1:MAX_ITER

        % x-update
        tmp = [ rho*eye(n), A'; A, zeros(m) ] \ [ rho*(z - u); b ];
        x = tmp(1:n);

        % z-update with relaxation
        zold = z;
        x_hat = alpha*x + (1 - alpha)*zold;
        z = pos(x_hat + u);

        u = u + (x_hat - z);
    end

end

function obj = objective(x)
    obj = x;
end
