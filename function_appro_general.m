clear all;
close all; 
clc;

a_limit=1000;
s_limit=200;
mu_limit=50;
sig_limit=10;
u_lb=-200;
u_ub=200;
N=200;

%%%%%%%%%%%%%%
a_pre=a_limit*rand(28,1);
%a_bar=a;
%ob=object(1,1,1,1,1,a);
A=[];
b=[];
x0=10;
V=zeros(N,1);
x=zeros(N,6);
%a0=ones(21,1);
tolerence=0.0000001;
change=1000;
j=0;
Er=[];
V_test=[];
k=1;
s_test=(-1+2*rand)*s_limit;
mu1_test=(-1+2*rand)*mu_limit;
mu2_test=(-1+2*rand)*mu_limit;
sig1_test=rand*sig_limit;
sig2_test=rand*sig_limit;
sig12_test=rand*sig_limit;
x_test=[s_test;mu1_test;mu2_test;sig1_test;sig2_test;sig12_test];

%%%%%%%%sample N hyperstates
x1=(-1+2*rand(N,1))*s_limit;
x23=(-1+2*rand(N,2))*mu_limit;
x456=rand(N,3)*sig_limit;
x=[x1 x23 x456];
%%%%%%%%%%%%
while j<10 %change>tolerence
    for i=1:N
        fun=@(u)object(x(i,:),u,a_pre);
        [u_opt,V(i)]=fmincon(fun,x0,A,b);
    end

fun2=@(a)error(V,x,a);
[a_opt,err]=fmincon(fun2,a_pre,A,b);
fun_test=@(u)object(x_test,u,a_opt);
[u_opt_test,V_test(k)]=fmincon(fun_test,x0,A,b);
change=norm(a_opt-a_pre);
a_pre=a_opt;
disp(change);
Er(k)=change;
if change<tolerence
    j=j+1;
end
figure(1);
plot(1:k,Er);
%ylim([0 10])
figure(2);
plot(1:k,V_test);
%ylim([0 10000])
k=k+1;


end



function err=error(V,x,a)
N=length(V);
err_v=zeros(N,1);
for i=1:N
    V_LHS=quadPoly(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),a);%Left hand side of Bellman Equation
    err_v(i)=V_LHS-V(i);
end
err=norm(err_v);
end


function ob=object(x,u,a) %compute the RHS of Bellman equation
%syms u;
%%%%%%%%%%%%parameters
Sigma=0.01;
R=10;
Q=1;
gamma=0.5;
%%%%%%%%%%
s=x(1);
b=x(2:6);
%%%%%%%%%%compute h_0,h_1,h_2
h0=computePoly_prime(0,b,s,u,Sigma,a); %s_prime=0
h0_h1_h2=computePoly_prime(1,b,s,u,Sigma,a); %s_prime=1:h0+h1+h2
h0_minusH1_h2=computePoly_prime(-1,b,s,u,Sigma,a); %s_prime=-1:h0-h1+h2
h2=(h0_h1_h2+h0_minusH1_h2)/2-h0;
h1=h0_h1_h2-h0-h2;
%%%%%%%%%%%%%%compute mean and cov
mean=b(1)*s+b(2)*u; %E[s' \mid s,u,b]
cov=b(3)*s^2+b(4)*u^2+2*b(5)*s*u+Sigma+mean^2;
%%%%%%%%%%%inte
integ=h0+h1*mean+h2*cov;
r=Q*s^2+R*u^2;
%%%%%%%%%%%%%%Bellman 
ob=r+gamma*integ;
end 

function poly_prime=computePoly_prime(s_prime,b,s,u,Sigma,a)
b_prime=beliefUpdate(b,s,u,s_prime,Sigma);
poly_prime=quadPoly(s_prime,b_prime(1),b_prime(2),b_prime(3),b_prime(4),b_prime(5),a);
end


function b_prime=beliefUpdate(b,s,u,s_prime,Sigma)
mu1=b(1);
mu2=b(2);
sig1=b(3);
sig2=b(4);
sig12=b(5);
G=computeG(sig1,sig2,sig12,s,u,Sigma);
Phi=[s u];
P_theta=[sig1 sig12;sig12 sig2];
mu=[mu1;mu2];
mu_prime=mu+G*(s_prime-Phi*mu);
mu1_prime=mu_prime(1);
mu2_prime=mu_prime(2);
P_theta_prime=P_theta-G*Phi*P_theta;
sig1_prime=P_theta_prime(1,1);
sig2_prime=P_theta_prime(2,2);
sig12_prime=P_theta_prime(2,1);
b_prime=[mu1_prime;mu2_prime;sig1_prime;sig2_prime;sig12_prime];
end


function G=computeG(sig1,sig2,sig12,s,u,Sigma)
P_theta=[sig1 sig12;sig12 sig2];
Phi=[s u];
G=P_theta*Phi.'*inv(Phi*P_theta*Phi.'+Sigma);
end


function poly=quadPoly(x1,x2,x3,x4,x5,x6,a)
%%%%%%%%assign a
a0=a(1); a1=a(2); a2=a(3);a3=a(4); a4=a(5); a5=a(6);a6=a(7);
a11=a(8); a12=a(9); a13=a(10);a14=a(11); a15=a(12);a16=a(13);
a22=a(14); a23=a(15); a24=a(16); a25=a(17); a26=a(18);a33=a(19); a34=a(20);a35=a(21);a36=a(22);
a44=a(23);a45=a(24);a46=a(25);a55=a(26);a56=a(27);a66=a(28);
x=[x1;x2;x3;x4;x5;x6];
first_order=[a1 a2 a3 a4 a5 a6]*x;
x_quad=[x1^2;x1*x2;x1*x3;x1*x4;x1*x5;x1*x6;x2^2;x2*x3;x2*x4;x2*x5;x2*x6;x3^2;x3*x4;x3*x5;x3*x6;x4^2;x4*x5;x4*x6;x5^2;x5*x6;x6^2];
second_order=[a11 a12 a13 a14 a15 a16 a22 a23 a24 a25 a26 a33 a34 a35 a36 a44 a45 a46 a55 a56 a66]*x_quad;
poly=a0+first_order+second_order;
end