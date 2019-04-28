% this code test the diffusion process in 2 dimension
% v(x) = cos(x1 + 2*x2)
clear
clc

num_sample = 256;
T = 0.5;
n = 5;
lambda=0;
num_interval = 5 * (2^n);
dt = T / num_interval;
sqrtdt = sqrt(dt);
t = 0:dt:T;
sigma = sqrt(2);
N=1000;
error = zeros(1,N);
%%
for m=1:N

dW1 = sqrtdt * randn(num_sample, num_interval);
dW2 = sqrtdt * randn(num_sample, num_interval);
X1 = zeros(num_sample, num_interval+1);
X2 = zeros(num_sample, num_interval+1);
u = zeros(num_sample, num_interval+1);
X1(:, 1) = rand(num_sample,1) * 2 * pi;
X2(:, 1) = rand(num_sample,1) * 2 * pi;
%%
for i=1:num_interval
    X1(:,i+1) = X1(:,i) - sin( X1(:,i)+ 2 * X2(:,i) ) * dt + sigma * dW1(:,i);
    X2(:,i+1) = X2(:,i) - 2 * sin(X1(:,i)+ 2 * X2(:,i) ) * dt + sigma * dW2(:,i);
end
%%
u(:,1) = exp(-cos(X1(:,1) + 2 * X2(:,1) ) );
for i=1:num_interval
    u(:,i+1) = u(:,i) + (5* cos(X1(:,i)+ 2 * X2(:,i) ) - lambda ) .* u(:,i) * dt...
        + sigma * sin(X1(:,i)+ 2 * X2(:,i) ) .* exp(-cos( X1(:,i)+ 2 * X2(:,i) ) ) .* dW1(:,i) ...
        + sigma * 2 * sin( X1(:,i)+ 2 * X2(:,i) ) .* exp(-cos( X1(:,i)+ 2 * X2(:,i) ) ) .* dW2(:,i);
end
u_T = exp(-cos(X1(:,end) + 2 * X2(:,end) ));
delta = u_T - u(:,end);
error(m) = norm(delta) / norm(u_T);
end
%end
a = mean(error);
%plot(a)