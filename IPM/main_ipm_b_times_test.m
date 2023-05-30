clear all
close all
clc

format short

ls_method = 'augmented'
gmres_mod = 'v1'
K = 100
n_vec = linspace(10^4, 10^5, 10);

for i=1:10
n = n_vec(i)




% Definition of the non-convex quadratic objective function
c = ones(n,1);
Q = spdiags([-c, 2*c, -c], -1:1, n,n);

% Definition of the equality constraint
e_K = ones(K,n/K);
A = spdiags(e_K, 0:K:(n-K), K, n);

b = ones(K, 1);

AA = sparse(2*K + n, n);
AA(1:K, 1:n) = A;
AA(K+1:2*K, 1:n) = -A;
AA(2*K+1:end, 1:n) = speye(n);

bb = [b;-b;zeros(n,1)];

A = AA; %2K + n
b = bb; %2K + n

% Choice of the factor tau_k
tau = @(mu) 0.3*exp(-mu)+0.70;

% Parameters for the convergence criterion
eps = 1.0e-12;
kmax = 100;

% Definition of the initial condition (satisfying the KKT conditions)
y0 = ones(2*K+n,1); % y = Ax-b
lambda0 = ones(2*K+n,1);
x0 = A\(b+y0); % <- least squares sense



fprintf("--------------Start IPM Predictor Corrector----\n");
% Predictor-Corrector Interior Point Method
tic
[xk, fk, yk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_B(...
    Q, c, A, b, tau, eps, kmax, x0, y0, lambda0, ls_method, gmres_mod);
times_vec(i) = toc;
%A*xk

% Error check of the KKT conditions
kkt1_err_linf = norm(Q*xk + c - A'*lambdak, inf);
kkt1_err_l2(i) = norm(Q*xk + c - A'*lambdak , 2);

kkt2_err_linf = norm(A*xk-b-yk, inf);
kkt2_err_l2 = norm(A*xk-b-yk, 2);

kkt3_prod    = yk'*lambdak;
kkt3_max_err = max(yk.*lambdak);

kkt4_ineq_x = all(yk < 0);
kkt4_ineq_s = any(lambdak < 0);

fprintf("--------------KKT condtion errors--------------\n");
fprintf(" Stationarity (2)\t\t\t\t\t=\t%.3e\n", kkt1_err_linf);
fprintf(" Stationarity (inf)\t\t\t\t\t=\t%.3e\n", kkt1_err_l2(i));
fprintf(" Primal Feasibility Equality (2)\t=\t%.3e\n", kkt2_err_l2);
fprintf(" Primal Feasibility Equality (inf)\t=\t%.3e\n", kkt2_err_linf);
fprintf(" Primal Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_x);
fprintf(" Dual Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_s);
fprintf( " Complementary Slackness (max)\t\t=\t%.3e\n", kkt3_max_err);
fprintf(" Complementary Slackness (prd)\t\t=\t%.3e\n\n\n\n", kkt3_prod);

%save("ipm_qp_results.mat")
end


yyaxis left
p1 = plot(1:10, times_vec, 'linewidth', 2, 'markersize', 3); hold on
title("Times and accuracy", 'interpreter', 'latex', 'FontSize', 14)
xticks(1:10)
%yticks(1:N+1)
%xticklabels(n_vec)
xticklabels(compose("%d*10^4", 1:10));
xlabel('$n$', 'interpreter', 'latex')
%ylabel('$[s]$', 'Interpreter','latex')

yyaxis right
p2 = semilogy(1:10, kkt1_err_l2,'linewidth', 2)

%semilogy(1:dmax, smooth(r_sim),'-', 'linewidth', 2)
%ylabel('KKT err $l_2$', 'Interpreter','latex', 'Rotation',45)
legend([p1, p2], {'Time $[s]$', 'Stationarity error $l_2$ norm'}, 'Location', 'northwest', 'Interpreter','latex')
grid on




