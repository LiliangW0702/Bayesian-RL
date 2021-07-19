clear all;
close all;
clc;
%parameters
Sigma=0.01;
Q=10;
R=1;
gamma=0.5;

%sample range
a_lim=10;
s_lim=20;
mu_lim=5;
sig_lim=1;
%sample data points for collocation
N=200;%number of data points
X=sample(N,s_lim,mu_lim,sig_lim);%sample N data points

tolerence=1e-7;
%initialization
change=1000;
load('a0.mat');
a_pre=a0;
%a_pre=a_lim*(-1+2*rand(21,1));
%a0=a_pre;
u0=0;
%sample a test point
x_test=[-9.2996 0.7628 0.2005 0.9789 0.6607 0.8462];%x_test=sampleTestpoint(s_lim,mu_lim,sig_lim);

Er=[];
V_test=[];
k=1;
j=0;
%quadPoly_test();

while j<10
    [V,u_RHS]=computeBellmanRHS(X,u0,Sigma,Q,R,gamma,a_pre);
    alpha=10;%regulation weight
    A=computeA(X);
    a_opt=linearRegression(A,V,alpha);
    
    fun_test=@(u)object(x_test,u,Sigma,Q,R,gamma,a_opt);
    [u_opt_test,V_test(k)]=fmincon(fun_test,u0,[],[]);
    change=norm(a_opt-a_pre);
    a_pre=a_opt;
    %plot
    Er(k)=log(change);
    if change<tolerence
        j=j+1;
    end
    %%%%print
    %disp()
    figure(1);
    plot(1:k,Er);
    %ylim([0 10])
    figure(2);
    plot(1:k,V_test);
    %ylim([0 10000])
    k=k+1;



end


function A=computeA(X) %compute the A in linear regression problem
N=length(X);
A=zeros(N,21);
for i=1:N
    x=X(i,:);
    x1=x(1);
    x2=x(2);
    x3=x(3);
    x4=x(4);
    x5=x(5);
    x_5=x(1:5);
    second_order_x=[x1^2 x1*x2 x1*x3 x1*x4 x1*x5 x2^2 x2*x3 x2*x4 x2*x5  x3^2 x3*x4 x3*x5 x4^2 x4*x5  x5^2 ];
    A(i,:)=[1 x_5 second_order_x];
end
end

function weights=linearRegression(A,b,alpha) %minimize norm(Ax-b)^2+norm(alpha*x)^2
%N=length(A);
den=transpose(A)*A+alpha^2*eye(21);
%disp(den);
weights=inv(den)*A.'*b;
end

function [V_RHS,u_RHS]=computeBellmanRHS(X,u0,Sigma,Q,R,gamma,a)
N=length(X);
V_RHS=zeros(N,1);
u_RHS=zeros(N,1);
for i=1:N
    x=X(i,:);
    fun=@(u)object(x,u,Sigma,Q,R,gamma,a);
    [u_RHS(i),V_RHS(i)]=fmincon(fun,u0,[],[],[],[],-200,200);
end
end

function object_test()
x=[2.5;-9.49700599;6.00598802;0.8003992;0.201596811;-0.3992016];
x=x.';
a=ones(21,1);
Sigma=0.01;
Q=10;
R=1;
gamma=0.5;
u=2;
target=[5.5861;0.0231;1.6577];
[~,h0,h1,h2]=object(x,u,Sigma,Q,R,gamma,a);
result=[h0;h1;h2];
disp(result);
assert (norm([h0;h1;h2]-target)^2<1e-7,'error in object function');
end



function ob=object(x,u,Sigma,Q,R,gamma,a) %compute the expression in [.] on RHS of Bellman equation
%compute h_0,h_1,h_2
h0=computePoly_prime(x,u,0,Sigma,a);
h0_h1_h2=computePoly_prime(x,u,1,Sigma,a);
h0_minus_h1_h2=computePoly_prime(x,u,-1,Sigma,a);
h2=(h0_h1_h2+h0_minus_h1_h2)/2-h0;
h1=h0_h1_h2-h0-h2;
%%%%%%%compute mean and second-order momentum
s=x(1);
mean=x(2)*s+x(3)*u;
Sec_moment=x(4)*s^2+x(5)*u^2+2*x(6)*s*u+Sigma+mean^2;
%integral
integ=h0+h1*mean+h2*Sec_moment;
r=Q*s^2+R*u^2;
ob=r+gamma*integ;
end

function computePoly_prime_test()
a=ones(21,1);
x=[1 -8 9 1 1 0];
u=2;
s_prime=2.5;
Sigma=0.01;
result=computePoly_prime(x,u,s_prime,Sigma,a);
target=67.6092;
assert((result-target)^2<1e-7,'error in computePoly_prime function');
end

function poly_prime=computePoly_prime(x,u,s_prime,Sigma,a)
b_prime=beliefUpdate(x,u,s_prime,Sigma);
x_prime=[s_prime;b_prime];
poly_prime=quadPoly(x_prime.',a);
end


function beliefUpdate_test()
x=[1 -8 9 1 1 0];
u=2;
s_prime=2.5;
Sigma=0.01;
result=beliefUpdate(x,u,s_prime,Sigma);
target=[-9.49700599;6.00598802;0.8003992;0.201596811;-0.3992016];
disp(result);
assert(norm(result-target)^2<1e-7,'error in beliefUpdate function');
end

function b_prime=beliefUpdate(x,u,s_prime,Sigma) %return a column vector b_prime
[G,P_theta,Phi]=computeG(x,u,Sigma);
x=x.';
mu=x(2:3);
mu_prime=mu+G*(s_prime-Phi*mu);
P_theta_prime=P_theta-G*Phi*P_theta;
b_prime=[mu_prime;P_theta_prime(1,1);P_theta_prime(2,2);P_theta_prime(1,2);];
end

function computeG_test()
x=[1 -8 9 1 1 0];
u=2;
Sigma=0.01;
[result,~,~]=computeG(x,u,Sigma);
disp(result);
assert ( norm(result-[0.1996008;0.3992016])^2< 1e-7,'error in computeG function');
end

function [G,P_theta,Phi]=computeG(x,u,Sigma)
P_theta=[x(4) x(6);x(6) x(5)];
Phi=[x(1) u];
K1=Phi*P_theta*Phi.'+Sigma;
G=P_theta*Phi.'*inv(K1);
end

function quadPoly_test()
%test 1
a=ones(21,1);
x=ones(1,6);
result=quadPoly(x,a);
assert ( (result-21)^2< 1e-7,'Wrong quadPoly function');
%%%%%%%test 2
a1=zeros(6,1);%zero_order,first_order
a2=2.0*ones(15,1);
a2(1)=1;
a2(6)=1;
a2(10)=1;
a2(13)=1;
a2(15)=1;
a=[a1;a2];
x=ones(1,6);
result=quadPoly(x,a);
assert ( (result-25)^2< 1e-7,'Wrong quadPoly function');
end


function poly=quadPoly(x,a) %input x is a row vector
x1=x(1);
x2=x(2);
x3=x(3);
x4=x(4);
x5=x(5);
x_5=x(1:5);
second_order_x=[x1^2;x1*x2;x1*x3;x1*x4;x1*x5;x2^2;x2*x3;x2*x4;x2*x5;x3^2;x3*x4;x3*x5;x4^2;x4*x5;x5^2];
poly=a.'*[1;x_5.';second_order_x];
end

function x_test=sampleTestpoint(s_limit,mu_limit,sig_limit)
s_test=(-1+2*rand)*s_limit;
mu1_test=(-1+2*rand)*mu_limit;
mu2_test=(-1+2*rand)*mu_limit;
sig1_test=rand*sig_limit;
sig2_test=rand*sig_limit;
sig12_test=rand*sig_limit;
x_test=[s_test mu1_test mu2_test sig1_test sig2_test sig12_test];
end


function X=sample(N,s_limit,mu_limit,sig_limit)
X1=(-1+2*rand(N,1))*s_limit;
X23=(-1+2*rand(N,2))*mu_limit;
X456=rand(N,3)*sig_limit;
X=[X1 X23 X456];
end