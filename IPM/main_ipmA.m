clear all
close all
clc

format short

fprintf("--------------Problem Initialization-----------\n");

% Initialization of the problem dimensions
n = 10^3;
K = 100;

% Definition of the non-convex quadratic objective function
c = ones(n,1);
Q = spdiags([-c, c, -c], -1:1, n,n);

% Definition of the equality constraint
e_K = ones(K,n/K);
A = spdiags(e_K, 0:K:(n-K), K, n);

b = ones(K, 1);

% Choice of the factor tau_k
tau = @(mu) 0.3*exp(-mu)+0.70;

% Parameters for the convergence criterion
eps = 1.0e-12;
kmax = 100;

% Definition of the initial condition (satisfying the KKT conditions)
x0 = ones(n,1);
s0 = ones(n,1);
lambda0 = A'\(-Q*x0 - c + s0);

fprintf("--------------Start IPM Predictor Corrector----\n");
% Predictor-Corrector Interior Point Method
[xk, fk, sk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_A(...
    Q, c, A, b, tau, eps, kmax, x0, lambda0, s0, 'augmented');
fprintf("--------------End IPM Predictor Corrector------\n");
%A*xk

th_sol = K/n*ones(n,1);

% Error check of the KKT conditions
kkt1_err_linf = norm(Q*xk + c + A'*lambdak -sk, inf);
kkt1_err_l2 = norm(Q*xk + c + A'*lambdak -sk, 2);

kkt2_err_linf = norm(A*xk-b, inf);
kkt2_err_l2 = norm(A*xk-b, 2);

kkt3_prod    = xk'*sk;
kkt3_max_err = max(xk.*sk);

kkt4_ineq_x = all(xk < 0);
kkt4_ineq_s = any(sk < 0);

fprintf("--------------KKT condtion errors--------------\n");
fprintf(" Stationarity (2)\t\t\t\t\t=\t%.3e\n", kkt1_err_linf);
fprintf(" Stationarity (inf)\t\t\t\t\t=\t%.3e\n", kkt1_err_l2);
fprintf(" Primal Feasibility Equality (2)\t=\t%.3e\n", kkt2_err_l2);
fprintf(" Primal Feasibility Equality (inf)\t=\t%.3e\n", kkt2_err_linf);
fprintf(" Primal Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_x);
fprintf(" Dual Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_s);
fprintf(" Complementary Slackness (max)\t\t=\t%.3e\n", kkt3_max_err);
fprintf(" Complementary Slackness (prd)\t\t=\t%.3e\n", kkt3_prod);

save("ipm_qp_results.mat")





