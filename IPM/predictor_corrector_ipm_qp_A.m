function [xk, fk, sk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_A(...
    Q, c, A, b, tau, eps, kmax, x0, lambda0, s0, ls_method)
%% Initialization of the method
[m, n] = size(A);
e = ones(n,1);

xk = x0;
lambdak = lambda0;
sk = s0;

mu0 = xk'*sk/n;
muk = mu0;
mu_seq = zeros(kmax,1);
tau_seq = zeros(kmax,1);
constr_err = zeros(kmax,1);
xs_seq=zeros(kmax,1);

for k = 1:kmax    
    %% Solve system 1 (predictor)
    
    r1 = -(Q*xk + c -sk + A'*lambdak);
    r2 = -(A*xk - b);
    r3 = -xk.*sk;
    
    D = spdiags(sk./xk, 0, n,n );
    switch ls_method
        case 'augmented'
            K = [Q + D, A';...
                A, zeros(m,m)];
            
            rtilde = [r1 + r3./xk;r2];
            Delta_augmented = gmres(K, rtilde, [], 1.0e-6, 20);
            
            Delta_xk_aff = Delta_augmented(1:n, 1);
            Delta_lambdak_aff = Delta_augmented(n+1:end, 1);
            Delta_sk_aff = r3./xk - sk.*Delta_xk_aff./xk;
        otherwise
    end
    %% Compute the affine step length in order to have positivity
    if any(Delta_xk_aff<0)
        alpha_P_aff = min(-xk(Delta_xk_aff<0)./Delta_xk_aff(Delta_xk_aff<0));
    else
        alpha_P_aff = 1;
    end
    
    if any(Delta_sk_aff<0)
        alpha_D_aff = min(-sk(Delta_sk_aff<0)./Delta_sk_aff(Delta_sk_aff<0));
    else
        alpha_D_aff = 1;
    end
    
    alpha_k_aff = min([1, alpha_P_aff, alpha_D_aff]);
    
    %% Compute the measure of centrality and the parameter
    muk_aff = ((xk + alpha_k_aff*Delta_xk_aff)'*(sk + alpha_k_aff*Delta_sk_aff))/n;
    sigmak = (muk_aff/ muk)^3;
    
    %% Solve system 1 (correction)
    r3 = r3 -Delta_xk_aff.*Delta_sk_aff + sigmak*muk;
    
    switch ls_method
        case 'augmented'          
            rtilde = [r1 + r3./xk;r2];
            Delta_augmented = gmres(K, rtilde, [], 1.0e-6, 20);
            
            Delta_xk = Delta_augmented(1:n, 1);
            Delta_lambdak = Delta_augmented(n+1:end, 1);
            Delta_sk = r3./xk - sk.*Delta_xk./xk;
            
        otherwise
    end
   
    
    %% Compute the affine step length in order to have positivity
    if any(Delta_xk<0)
        alpha_P = min(-tau(muk)*xk(Delta_xk<0)./Delta_xk(Delta_xk<0));
    else
        alpha_P = 1;
    end
    if any(Delta_sk<0)
        alpha_D = min(-tau(muk)*sk(Delta_sk<0)./Delta_sk(Delta_sk<0));
    else
        alpha_D = 1;
    end
    
    alpha_k = min([1, alpha_P, alpha_D]);
    
    %% Exit check
    muk = ((xk + alpha_k*Delta_xk)'*(sk + alpha_k*Delta_sk))/n;
    mu_seq(k) = muk;
    tau_seq(k) = tau(muk);
    if muk <= eps*mu0
        break;
    end

    fkseq(k) = 0.5*xk'*Q*xk + c'*xk;
    constr_err(k) = norm(A*xk-b);
    xs_seq(k)=xk'*sk;
    
    %% Update
    xk = xk + alpha_k*Delta_xk;
    lambdak = lambdak + alpha_k*Delta_lambdak;
    sk = sk + alpha_k*Delta_sk;
       
end
    tau_seq = tau_seq(1:k);
    
    %% Evaluate the function f in xk
    fk = 0.5*xk'*Q*xk + c'*xk;
       
end