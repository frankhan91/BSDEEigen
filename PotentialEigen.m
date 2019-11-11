% This code compute the eigenvalue and eigenfunction of the operator -\Delta +
% \sum_{i=1}^{d} c_i * cos(x_i) via spectrum method
clear
clc
%tic
d=20;
N=20;
%c = rand([1,d]);
% c = [0.814723686393179,0.905791937075619,0.126986816293506,...
%     0.913375856139019,0.632359246225410,0.097540404999410,...
%     0.278498218867048,0.546881519204984,0.957506835434298,0.964888535199277];
c = [0.814723686393179,0.905791937075619,0.126986816293506,0.913375856139019,...
    0.632359246225410,0.0975404049994095,0.278498218867048,0.546881519204984,...
    0.957506835434298,0.964888535199277,0.157613081677548,0.970592781760616,...
    0.957166948242946,0.485375648722841,0.800280468888800,0.141886338627215,...
    0.421761282626275,0.915735525189067,0.792207329559554,0.959492426392903];
%c = c .* [1, 0.01 * ones(1,9)];
c = c * 0.1; %true_eigen = -0.203549513655507 for 20dim

coef = zeros(d,10);
grad_coef = zeros(d,10);
eigeni = zeros(1,d);
mm = 0:9;
for i=1:d
a = 1:N; a = a.^2;
b = [a(end:-1:1), 0, a];
e = c(i) *diag(ones(2*N,1),1);
A = diag(2*b) + e + e';
[V,D] = eig(A);
eigeni(i) = D(1,1)/2 ;
temp = V(:,1)';
temp = [temp(N+1), 2*temp(N+2:2*N+1)];
if temp(1) < 0
    temp = -temp;
end
coef(i,:) = temp(1:10);
grad_coef(i,:) = temp(1:10) .* mm;
end
coef = coef';
grad_coef = grad_coef';
eigen = sum(eigeni);
%eigen1234 = [eigeni;eigen2i;eigen3i;eigen4i];
%save('eigen_and_coef.m', 'eigen', 'eigeni', 'coef', 'grad_coef','-ascii','%.15')
%save('coef.txt','coef','-ascii')
%% check the histogram of eigenfunction
x = 2*pi* rand(10000,d);
base_cos = 0*x;
for m=1:10
    base_cos = base_cos + cos((m-1)*x) .* repmat(coef(m,:),10000,1);
end
y = prod(base_cos,2);
histogram(y)

%% output the coefficients

fileID = fopen('coef.txt','w');
fprintf(fileID,'self.coef=[');
for m=1:10
    fprintf(fileID,'[');
    for i=1:10
        fprintf(fileID,'%.14e,',coef(m,i));
    end
    fprintf(fileID,'\n');
    for i=11:19
        fprintf(fileID,'%.14e,',coef(m,i));
    end
    fprintf(fileID,'%.14e]',coef(m,20));
    if m==10
        fprintf(fileID,']\n');
    else
        fprintf(fileID,',');
    end
    fprintf(fileID,'\n');
end
fprintf(fileID,'self.coef2=[');
for m=1:10
    fprintf(fileID,'[');
    for i=1:10
        fprintf(fileID,'%.14e,',grad_coef(m,i));
    end
    fprintf(fileID,'\n');
    for i=11:19
        fprintf(fileID,'%.14e,',grad_coef(m,i));
    end
    fprintf(fileID,'%.14e]',grad_coef(m,20));
    if m==10
        fprintf(fileID,']');
    else
        fprintf(fileID,',');
    end
    fprintf(fileID,'\n');
end
fclose(fileID);

%%
%for second smallest eigenvalue in 2d smallest one in dim1 and second one
%in dim2
%N = 20; % truncation
a = 1:N; a = a.^2;
b = [a(end:-1:1), 0, a];
e = c(1) *diag(ones(2*N,1),1);
A = diag(2*b) + e + e';
[V,D] = eig(A);
eigen2 = D(2,2)/2;
temp = V(:,2)';%second smallest eigenvalue's eigenvector
temp = temp(N+2:2*N+1); %temp(N+1)=0 for second smallest eigenvalue's eigenvector
temp = temp(1:9);
if temp(1) < 0
    temp = -temp;
end
coef_second = [0,temp]';
coef_second_grad = [0,temp] .* (0:9); %row vector
coef_second_grad = coef_second_grad'; %column vector

%%
NN = 1000;
x = 0:(2*pi/NN):2*pi;
sup = zeros(1,d);
for j = 1:d
    y = x*0;
    for m = 1:10
        y = y + coef(m,j) * cos((m-1)*x);
    end
    sup(j) = max(abs(y));
end

%% diffusion 
% to test the error of diffusion process
num_sample = 256;
T = 0.03;
n = 6;
lambda = eigen;
num_interval = 5 * (2^n);
dt = T / num_interval;
sqrtdt = sqrt(dt);
sigma = sqrt(2);
M=10;
error = zeros(1,M);

for index=1:M
dW = sqrtdt * randn(num_sample, d, num_interval);
X = zeros(num_sample, d, num_interval+1);
u = ones(num_sample, num_interval+1);
X(:,:,1) = reshape( rand(num_sample,d) * 2 * pi , [num_sample,d,1]);

%
for i=1:num_interval
    X(:,:,i+1) = X(:,:,i) + sigma * dW(:,:,i);
end
%
for j = 1:10
    u(:,1) = u(:,1) .* ( cos( (j-1) * reshape(X(:,:,1),[num_sample,d]) ) * coef(:,j) );
end
for i=1:num_interval
    %grad_ui = ones(num_sample,d);
    base_cosxi = ones(num_sample,d);
    base_sinmxi = ones(num_sample,d);
    for j = 1:10
        base_cosxi(:,j) = cos( (j-1) * reshape(X(:,:,i),[num_sample,d]) ) * coef(:,j);
        base_sinmxi(:,j) = -sin( (j-1) * reshape(X(:,:,i),[num_sample,d]) ) * grad_coef(:,j);
    end
    prodcos = prod(base_cosxi,2);
    grad_ui = repmat(prodcos,[1,d]);
    grad_ui = grad_ui .* base_sinmxi ./ base_cosxi;
    u(:,i+1) = u(:,i) + (cos( reshape(X(:,:,i), [num_sample,d]) ) * c' - lambda ) .* u(:,i) * dt...
        + sigma * sum( grad_ui .* reshape( dW(:,:,i) , [num_sample,d]) ,2 );
end
u_T = ones(num_sample,1);
for j = 1:10
    u_T = u_T .* ( cos( (j-1) * reshape(X(:,:,end),[num_sample,d]) ) * coef(:,j) );
end
delta = u_T - u(:,end);
error(index) = norm(delta) / norm(u_T);
end
%end
a = mean(error);
%t=toc;

%% this section test the init_rel_loss of NN
clear
clc
true_y = importdata('C:\Users\mozho\Mo Zhou\Eigen\BSDEEigen-master\logs\SchrodingerHistTruey.txt');
init_y = importdata('C:\Users\mozho\Mo Zhou\Eigen\BSDEEigen-master\logs\SchrodingerHistNN.txt');
x=1:1000;
error = true_y - init_y;
rel_err = sqrt(mean(error.^2));


[0.0814723686393179,0.0905791937075619,0.0126986816293506,0.0913375856139019,0.0632359246225410,0.00975404049994095,0.0278498218867048,0.0546881519204984,0.0957506835434298,0.0964888535199277,0.0157613081677548,0.0970592781760616,0.0957166948242946,0.0485375648722841,0.0800280468888800,0.0141886338627215,0.0421761282626275,0.0915735525189067,0.0792207329559554,0.0959492426392903]

[0.162944737278636,0.181158387415124,0.0253973632587012,0.182675171227804,0.126471849245082,0.0195080809998819,0.0556996437734096,0.109376303840997,0.191501367086860,0.192977707039855,0.0315226163355096,0.194118556352123,0.191433389648589,0.0970751297445682,0.160056093777760,0.0283772677254430,0.0843522565252550,0.183147105037813,0.158441465911911,0.191898485278581]