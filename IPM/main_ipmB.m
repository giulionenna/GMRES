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

ls_method = 'augmented';
%ls_method = 'substitution';

gmres_mod = 'v1';

fprintf("--------------Start IPM Predictor Corrector----\n");
% Predictor-Corrector Interior Point Method
[xk, fk, yk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_B(...
    Q, c, A, b, tau, eps, kmax, x0, y0, lambda0, ls_method, gmres_mod);
fprintf("--------------End IPM Predictor Corrector------\n");
%A*xk

% Error check of the KKT conditions
kkt1_err_linf = norm(Q*xk + c - A'*lambdak, inf);
kkt1_err_l2 = norm(Q*xk + c - A'*lambdak , 2);

kkt2_err_linf = norm(A*xk-b-yk, inf);
kkt2_err_l2 = norm(A*xk-b-yk, 2);

kkt3_prod    = yk'*lambdak;
kkt3_max_err = max(yk.*lambdak);

kkt4_ineq_x = all(yk < 0);
kkt4_ineq_s = any(lambdak < 0);

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





