clear
close all
clc

rng(420);

N = 100;

A = 10*rand(N,N)-5;
b = 10*rand(N,1)-5;
x_0 = 1+zeros(N,1);

tol = 1e-15;
maxiter = 100;

[x, res] = gmres_v1(A,b, x_0, tol, maxiter);

x_true = A\b;

err = norm(x-x_true)/norm(x_true)

plot([x, x_true])