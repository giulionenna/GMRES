function [x, res] = gmres_v1(A,b, x_0, tol, maxiter)
%GMRES_V1 Summary of this function goes here
%   Detailed explanation goes here
n = length(A);
r_0 = b-A*x_0; %starting residual
beta = norm(r_0); %norm of the  first residual
res = beta; %norm of the residual
V = r_0/beta; %first element of the basis of the krilov space
m = 1;
R = (A*V)'*V;
Q = eye(1);
b_norm = norm(b);
g = beta;

while (res/b_norm >= tol && m<=maxiter)
    
    h = zeros(m+1, 1); %last column of current H_bar
    omega = A*V(:,end); %will be the new element of the basis
    for j=1:m %GS orthonormalization
        h(j) = omega'*V(:,j);
        omega = omega - h(j)*V(:,j);
    end
    h(m+1) = norm(omega);
    if abs(h(m+1)) <= tol 
        break %krilov space max dimension reached
    end
    
    %increase size of V and append last column
    V_new = zeros(n , m+1);
    V_new(:,1:m) = V;
    V_new(:,m+1) = omega/h(m+1);
    V = V_new;
    
    %increase size of Q and put 1 in last element of diagonal
    Q_new = zeros(m+1, m+1);
    Q_new(1:m, 1:m) = Q;
    Q_new(m+1, m+1) = 1;
    Q = Q_new;

    %increase size of R and append Qh in last column
    R_new = zeros(m+1, m);
    R_new(1:end-1, 1:end-1) = R;
    R_new(:,m) = Q*h;
    R = R_new;

    %compute givens rotation to eliminate R(m+1,m)
    G = eye(m+1, m+1); %TODO SPALLOC
    tmp = sqrt(R(m,m)^2 + R(m+1,m)^2);
    G(m,m) = 0;
    G(m+1,m+1) = 0;
    G(m,m) = R(m,m)/tmp;
    G(m,m+1) = R(m+1,m) /tmp;
    G(m+1,m) = - G(m,m+1);
    G(m+1, m+1) = G(m,m);

    %eliminate R(m+1,m)
    R = G*R;

    %store the rotation
    Q = G*Q;

    %compute the new g
    g_new = zeros(m+1,1);
    g_new(1:end-1) = g;
    g = g_new;
    g = G*g;

    res =abs(g(end));
    m = m+1;
end

y = R(1:end-1, :)\g(1:end-1); %TODO ASSICURARSI CHE CAPISCA CHE SIA TRIANGOLARE
x = x_0 + V(:, 1:end-1)*y;

end

