function [x,res, flag] = d_lanczos(A,b, x_0, tol, maxiter)
%D_LANCZOS Summary of this function goes here
%   Detailed explanation goes here
n = length(A);
r_0 = b-A*x_0;
x = x_0;
N = min(n, maxiter);
beta = zeros(N,1);
eta = zeros(N,1);
alpha = zeros(N,1);
lambda = zeros(N,1);
z = zeros(N,1);
P = zeros(n,N);

res = norm(r_0);
V = zeros(n, N);
V(:,1) = r_0/res;

beta(1) = res;
z(1) = beta(1);
eta(1) = (A*V(:,1))'*V(:,1);
alpha(1) = eta(1);

omega = A*V(:,1) - alpha(1)*V(:,1);
beta(2) = norm(omega);
V(:,2) = omega/beta(2);
P(:,1) = V(:,1)/eta(1);

m=1;
b_norm = norm(b);
while res/b_norm >=tol && m<=maxiter
    m = m+1;
    omega = A*V(:,m)- beta(m)*V(:,m-1);
    alpha(m) = omega'*V(:,m);
    omega = omega - alpha(m)*V(:,m);
    beta(m+1) = norm(omega);
    V(:,m+1) = omega/beta(m+1);
    lambda(m) = beta(m)/eta(m-1);
    eta(m) = alpha(m)-lambda(m)*beta(m);
    z(m) = -lambda(m)*z(m-1);
    P(:,m) = (1/eta(m))*(V(:,m) - beta(m)* P(:,m-1));
    x = x + z(m)*V(:, m);

    res = abs(beta(m+1)*z(m)/eta(m));
end
if m >= maxiter
    flag = 1;
else
    flag = 0;
end

end

