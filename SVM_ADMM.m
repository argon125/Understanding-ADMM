clc;clear all;close all;
t=linspace(-5,5,1000);
x=abs(t)+(t-10).^2;
A=[-t.*x -x];
linear_svm(A, 1, 10, 2, 1.2)

function [xave] = linear_svm(A, lambda, p, rho, alpha)
% linear_svm   Solve linear support vector machine (SVM) via ADMM
%
% Solves the following problem via ADMM:
%   minimize   (1/2)||w||_2^2 + \lambda sum h_j(w, b)
%
% where A is a matrix given by [-y_j*x_j -y_j], lambda is a
% regularization parameter, and p is a partition of the observations in toq
% different subsystems.
%
% This function implements a *distributed* SVM that runs its updates
% serially.
        
    MAX_ITER = 1000;%max iterations
    N=length(A)
    %ADMM solver
    x = zeros(1,N);%store elements of main variable
    z = zeros(1,N);%store elements of surrogate variable
    u = zeros(1,N);%store elements of Lagrangian

    for k = 1:MAX_ITER
        % x-update
        for i = 1:N % run upto max partition    %I removed comma here
            cvx_begin quiet
                variable x_var(N)
                minimize ( sum(pos(A*x_var + 1)) + rho/2*sum_square(x_var - z(:,i) + u(:,i)) )
            cvx_end
            x(:,i) = x_var;
        end
        xave = mean(x,2);%2D mean of matrix

        % z-update with relaxation
        zold = z;
        x_hat = alpha*x +(1-alpha)*zold;
        z = N*rho/(1/lambda + N*rho)*mean( x_hat + u, 2 );
        z = z*ones(1,N);

        % u-update
        u = u + (x_hat - z);
    end
end

function val = hinge_loss(A,x)
    val = 0;
    for i = 1:length(A)
        val = val + sum(pos(A{i}*x(:,i) + 1));
    end
end