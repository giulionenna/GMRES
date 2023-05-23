clear
close all
clc

rng(420);

N = 1000;

A = 10*sprand(N, N, 0.01);
b = 10*rand(N,1)-5;
x_0 = 1+zeros(N,1);

tol = 1e-14;
maxiter = 100;
restart = 200;
max_restart = 200;

[x_v1, res_v1, flag_v1] = gmres_v1(A,b, x_0, tol, maxiter);
[x_v2, res_v2, flag_v2, iter] = gmres_v2(A,b, x_0, tol, restart, max_restart);
%x_gmres = gmres(A, b, restart, tol, max_restart);

%x_true = A\b;

%err_v1 = norm(x_v1-x_gmres)/norm(x_gmres)
%err_v2 = norm(x_v2-x_gmres)/norm(x_gmres)
err_res_v1 = norm(A*x_v1-b)/norm(b)
err_res_v2 = norm(A*x_v2-b)/norm(b)

%plot([x_v1, x_v2, x_gmres])