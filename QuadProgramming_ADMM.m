clc;clear all;close all;
% quadprog  Solve standard form box-constrained QP via ADMM
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*x'*P*x + q'*x + r
%   subject to   lb <= x <= ub
%
% The solution is returned in the vector x.
% rho is the augmented Lagrangian parameter and alpha is the over-relaxation parameter
randn('state', 0);
rand('state', 0);

n = 5;

% generate a well-conditioned positive definite matrix
% (for faster convergence)
P = rand(n);
P = P + P';
[V D] = eig(P);
P = V*diag(1+rand(n,1))*V';

q = randn(n,1);
r = randn(1);

l = randn(n,1);
u = randn(n,1);
lb = min(l,u);
ub = max(l,u);

[x history] = quadprog(P, q, r, lb, ub, 1.0, 1.0);

function [z, history] = quadprog(P, q, r, lb, ub, rho, alpha)
    QUIET    = 0;
    MAX_ITER = 1000;
    ABSTOL   = 1e-4;
    RELTOL   = 1e-2;
    n = size(P,1);
    x = zeros(n,1);
    z = zeros(n,1);
    u = zeros(n,1);

    if ~QUIET
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
          'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end

    for k = 1:MAX_ITER

        if k > 1
            x = R \ (R' \ (rho*(z - u) - q));
        else
            R = chol(P + rho*eye(n));
            x = R \ (R' \ (rho*(z - u) - q));
        end

        % z-update with relaxation
        zold = z;
        x_hat = alpha*x +(1-alpha)*zold;
        z = min(ub, max(lb, x_hat + u));

        % u-update
        u = u + (x_hat - z);

        % diagnostics, reporting, termination checks
        history.objval(k)  = objective(P, q, r, x);

        history.r_norm(k)  = norm(x - z);
        history.s_norm(k)  = norm(-rho*(z - zold));

        history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
        history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

        if ~QUIET
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
                history.r_norm(k), history.eps_pri(k), ...
                history.s_norm(k), history.eps_dual(k), history.objval(k));
        end

        if (history.r_norm(k) < history.eps_pri(k) && ...
           history.s_norm(k) < history.eps_dual(k))
             break;
        end
    end
end

function obj = objective(P, q, r, x)
    obj = 0.5*x'*P*x + q'*x + r;
end
