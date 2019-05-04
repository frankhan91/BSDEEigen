% This code compute the eigenvalue and eigenfunction of the operator -D^2 +
% cos(x) in 1-D space via spectrum method
clear
clc

N = 20; % truncation
a = 1:N; a = a.^2;
b = [a(end:-1:1), 0, a];
e = diag(ones(2*N,1),1);
A = diag(2*b) + e + e';
[V,D] = eig(A);
eigen = D(1,1)/2 ;
coef = V(:,1)';
coef = [coef(N+1), 2*coef(N+2:2*N+1)];
