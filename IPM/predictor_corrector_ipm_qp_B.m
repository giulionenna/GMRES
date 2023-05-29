function [xk, fk, yk, lambdak, muk, k, fkseq, tau_seq] = predictor_corrector_ipm_qp_B(...
    Q, c, A, b, tau, eps, kmax, x0, y0, lambda0, ls_method, gmres_mod)
%% Initialization of the method
[m, n] = size(A);
e = ones(m,1);

xk = x0;
yk = y0;
lambdak = lambda0;

mu0 = yk'*lambdak/n;
muk = mu0;

mu_seq = zeros(kmax,1);
tau_seq = zeros(kmax,1);
constr_err = zeros(kmax,1);
xs_seq=zeros(kmax,1);

for k = 1:kmax    
    %% Solve system 1 (predictor)
    
    r1 = -(Q*xk + c - A'*lambdak);
    r2 = -(A*xk - b - yk);
    r3 = - yk.*lambdak;
    
    Dinv = spdiags(yk./lambdak, 0, m, m);
    D = spdiags(lambdak./yk, 0, m, m);
    switch ls_method
        case 'augmented'
            K = [Q, -A';...
                 A, Dinv];
            rtilde = [r1;r2 + r3./lambdak];
            
            if strcmp(gmres_mod, 'native')
                Delta_augmented = gmres(K, rtilde, [], 1.0e-6, 20);
            elseif strcmp(gmres_mod, 'v1')
                Delta_augmented = gmres_v1(K, rtilde, zeros(length(rtilde), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'none')
                 Delta_augmented = K\rtilde;
            elseif strcmp(gmres_mod, 'lanczos')
                 Delta_augmented = d_lanczos(K, rtilde, zeros(length(rtilde), 1), 1e-6, 20);
            end

            Delta_xk_aff = Delta_augmented(1:n, 1);
            Delta_lambdak_aff = Delta_augmented(n+1:end, 1);
            
            Delta_yk_aff = r3./lambdak - Delta_lambdak_aff.*yk./lambdak;
        case 'substitution'

            if strcmp(gmres_mod, 'native')
                Delta_xk_aff = pcg((Q + A'*D*A),( r1 + A'*(r2.*lambdak./yk + r3./yk)), 1.0e-6);
            elseif strcmp(gmres_mod, 'v1')
                 b_tmp=(r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk_aff = gmres_v1((Q + A'*D*A),b_tmp,  zeros(length(b_tmp), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'lanczos')
                b_tmp=(r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk_aff = d_lanczos((Q + A'*D*A),b_tmp,  zeros(length(b_tmp), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'none')
                b_tmp=(r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk_aff = (Q + A'*D*A)\b_tmp;
            end

            Delta_lambdak_aff = r2.*lambdak./yk + r3./yk - (A*Delta_xk_aff).*lambdak./yk;
            Delta_yk_aff = r3./lambdak - Delta_lambdak_aff.*yk./lambdak;
    end
    %% Compute the affine step length in order to have positivity
    if any(Delta_yk_aff<0)
        alpha_P_aff = min(-yk(Delta_yk_aff<0)./Delta_yk_aff(Delta_yk_aff<0));
    else
        alpha_P_aff = 1;
    end
    
    if any(Delta_lambdak_aff<0)
        alpha_D_aff = min(-lambdak(Delta_lambdak_aff<0)./Delta_lambdak_aff(Delta_lambdak_aff<0));
    else
        alpha_D_aff = 1;
    end
    
    alpha_k_aff = min([1, alpha_P_aff, alpha_D_aff]);
    
    %% Compute the measure of centrality and the parameter
    muk_aff = ((yk + alpha_k_aff*Delta_yk_aff)'*(lambdak + alpha_k_aff*Delta_lambdak_aff))/n;
    sigmak = (muk_aff/ muk)^3;
    
    %% Solve system 1 (correction)
    % Delta_Yk_aff = spdiags(Delta_yk_aff, 0, m,m);
    % Delta_LAMBDAk_aff = spdiags(Delta_lambdak_aff, 0, m,m);            
   
    r3 = r3 - Delta_yk_aff.*Delta_lambdak_aff + sigmak*muk;

    switch ls_method
        case 'augmented'
            rtilde = [r1;r2 + r3./lambdak];
            
            if strcmp(gmres_mod, 'native')
                Delta_augmented = gmres(K, rtilde, [], 1.0e-6, 20);
            elseif strcmp(gmres_mod, 'v1')
                Delta_augmented = gmres_v1(K, rtilde, zeros(length(rtilde), 1), 1e-6, 20);
            elseif strcmp(gmres_mod,'lanczos')
                Delta_augmented = d_lanczos(K, rtilde, zeros(length(rtilde), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'none')
                 Delta_augmented = K\rtilde;                
            end


            Delta_xk = Delta_augmented(1:n, 1);
            Delta_lambdak = Delta_augmented(n+1:end, 1);
            
            Delta_yk = r3./lambdak - Delta_lambdak.*yk./lambdak;
        case 'substitution'
            
            if strcmp(gmres_mod, 'native')
                Delta_xk = pcg((Q + A'*D*A),( r1 + A'*(r2.*lambdak./yk + r3./yk)), 1.0e-6);
            elseif strcmp(gmres_mod, 'v1')
                b_tmp = (r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk = gmres_v1((Q + A'*D*A),b_tmp,  zeros(length(b_tmp), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'lanczos')
               b_tmp = (r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk = d_lanczos((Q + A'*D*A),b_tmp,  zeros(length(b_tmp), 1), 1e-6, 20);
            elseif strcmp(gmres_mod, 'none')
                b_tmp = (r1 + A'*(r2.*lambdak./yk + r3./yk));
                Delta_xk = (Q + A'*D*A)\b_tmp;
            end
            
            Delta_lambdak = r2.*lambdak./yk + r3./yk - (A*Delta_xk).*lambdak./yk;
            Delta_yk= r3./lambdak - Delta_lambdak.*yk./lambdak;
    end

    %% Compute the affine step length in order to have positivity
    if any(Delta_yk<0)
        alpha_P = min(-tau(muk)*yk(Delta_yk<0)./Delta_yk(Delta_yk<0));
    else
        alpha_P = 1;
    end
    if any(Delta_lambdak<0)
        alpha_D = min(-tau(muk)*lambdak(Delta_lambdak<0)./Delta_lambdak(Delta_lambdak<0));
    else
        alpha_D = 1;
    end
    
    alpha_k = min([1, alpha_P, alpha_D]);
    
    %% Exit check
    muk = ((lambdak + alpha_k*Delta_lambdak)'*(yk + alpha_k*Delta_yk))/n;
    mu_seq(k) = muk;
    tau_seq(k) = tau(muk);
    if muk <= eps*mu0
        break;
    end
    
    fkseq(k) = 0.5*xk'*Q*xk + c'*xk;
    constr_err(k) = norm(yk);
    xs_seq(k)=lambdak'*yk;
    
    %% Update
    xk = xk + alpha_k*Delta_xk;
    lambdak = lambdak + alpha_k*Delta_lambdak;
    yk = yk + alpha_k*Delta_yk;
       
end
    tau_seq = tau_seq(1:k); 
    %% Evaluate the function f in xk
    fk = 0.5*xk'*Q*xk + c'*xk;
end