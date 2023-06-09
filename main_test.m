clear
close all
clc

rng(420);

N = 10000;

A = 10*sprand(N, N, 0.0001);
%A = speye(N);
%A = spdiags(100*rand(N,1), 1, N,N);
b = A*ones(N,1);
x_0 = zeros(N,1);

tol = 1e-8;
maxiter = 100;
restart = 30;
max_restart = 100;

[x_v1, res_v1, flag_v1] = gmres_v1(A,b, x_0, tol, maxiter);
[x_v2, res_v2, flag_v2, iter] = gmres_v2(A,b, x_0, tol, restart, max_restart);
x_gmres = gmres(A, b, restart, tol, max_restart);

err_v1 = norm(x_v1 - ones(N,1))/norm(ones(N,1))
err_v2 =  norm(x_v2 - ones(N,1))/norm(ones(N,1))
err_gmres = norm(x_gmres - ones(N,1))/norm(ones(N,1))
%err_v1 = norm(x_v1-x_gmres)/norm(x_gmres)
%err_v2 = norm(x_v2-x_gmres)/norm(x_gmres)
%err_res_v1 = norm(A*x_v1-b)/norm(b)
%err_res_v2 = norm(A*x_v2-b)/norm(b)


B = A*A';
b = B*ones(N,1);

[x_d,res_d, flag_d] = d_lanczos(B,b, x_0, tol, 10);

err_d = norm(x_d -ones(N,1))/norm(ones(N,1))


%plot([x_v1, x_v2, x_d])