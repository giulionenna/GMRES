clear all
close all
clc

format short

ls_method_vec ={'augmented', 'substitution'};
gmres_mod_vec = {'v1', 'native'};
K_vec = [100, 500];
n_vec = [3, 4, 5];
fid = fopen('log.txt', 'w');

for i = [1:2]
    for j = [1:2]
        for k_iter=[1:2]
            for h = [1:3]
fprintf(fid, "--------------Problem Initialization-----------\n");
fprintf(fid,"Running with params:\n");
fprintf(fid,"\t ls_method: \t %s \n", ls_method_vec{i});
ls_method = ls_method_vec{i};
fprintf(fid,"\t gmres_mod: \t %s \n", gmres_mod_vec{j});
gmres_mod = gmres_mod_vec{j};
fprintf(fid,"\t K: \t\t\t %i \n", K_vec(k_iter));
K = K_vec(k_iter);
fprintf(fid,"\t n: \t\t\t 10^%i \n", n_vec(h));
n = 10^n_vec(h);




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



fprintf(fid,"--------------Start IPM Predictor Corrector----\n");
% Predictor-Corrector Interior Point Method
tic
[xk, fk, yk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_B(...
    Q, c, A, b, tau, eps, kmax, x0, y0, lambda0, ls_method, gmres_mod);
fprintf(fid,"--------------End IPM Predictor Corrector------\n");
fprintf(fid,"Time elapsed\t\t\t\t\t\t=\t%.3e sec \n", toc);
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

fprintf(fid,"--------------KKT condtion errors--------------\n");
fprintf(fid," Stationarity (2)\t\t\t\t\t=\t%.3e\n", kkt1_err_linf);
fprintf(fid," Stationarity (inf)\t\t\t\t\t=\t%.3e\n", kkt1_err_l2);
fprintf(fid," Primal Feasibility Equality (2)\t=\t%.3e\n", kkt2_err_l2);
fprintf(fid," Primal Feasibility Equality (inf)\t=\t%.3e\n", kkt2_err_linf);
fprintf(fid," Primal Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_x);
fprintf(fid," Dual Feasibility Inequality\t\t=\t%d\n", kkt4_ineq_s);
fprintf(fid, " Complementary Slackness (max)\t\t=\t%.3e\n", kkt3_max_err);
fprintf(fid," Complementary Slackness (prd)\t\t=\t%.3e\n\n\n\n", kkt3_prod);

%save("ipm_qp_results.mat")
            end
        end
    end
end
fclose(fid)





