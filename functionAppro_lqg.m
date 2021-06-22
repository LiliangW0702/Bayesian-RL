clear all;
close all; 
clc;



a_limit=1000;
s_limit=200;
mu_limit=50;
sig_limit=10;
u_lb=-200;
u_ub=200;
N=2000;

%%%%%%%%%%%%%%
a_pre=a_limit*rand(3,1);
%a_bar=a;
%ob=object(1,1,1,1,1,a);
A=[];
b=[];
x0=10;
V=zeros(N,1);
x=zeros(N,1);
%a0=ones(21,1);
tolerence=0.01;
change=1000;
j=0;
Er=[];
V_test=[];
k=1;
s_test=4.0;


%%%%%%%%sample N states
for i=1:N
    s=(-1+2*rand)*s_limit;
    x(i)=s;
end
%%%%%%%%%%%%
while j<10 %change>tolerence
    for i=1:N
        s=x(i);
        u_opt=optCtrl(s,a_pre);
        V(i)=computeV(s,u_opt,a_pre);
    end

fun2=@(a)error(V,x,a);
a0=a_limit*rand(3,1);
[a_opt,err]=fmincon(fun2,a_pre,A,b);
u_opt_test=optCtrl(s_test,a_opt);
V_test(k)=computeV(s_test,u_opt_test,a_opt);
change=norm(a_opt-a_pre)/norm(a_pre);
a_pre=a_opt;
disp(change);
Er(k)=change;
if change<tolerence
    j=j+1;
end
figure(1);
plot(1:k,Er);
ylim([0 10])
figure(2);
plot(1:k,V_test);
%ylim([0 10000])
k=k+1;


end



function err=error(V,x,a)
N=2000;
for i=1:N
    err_v=zeros(N,1);
    V_next=value_function(x(i),a);
    err_v(i)=V_next-V(i);
    err=norm(err_v);
end
end


function V=value_function(s,a)
%%%%%%%%assign a
a0=a(1); a1=a(2); a2=a(3);
V=a0+a1*s+a2*s^2;
end

function ob=computeV(s,u,a)
%syms u;
%%%%%%%%%%%%parameters
Sigma=0.01;
R=10;
Q=1;
gamma=0.5;
theta1=-3;
theta2=2;
%%%%%%%%assign a
a0=a(1); a1=a(2); a2=a(3);
%%%%%%%%%%%inte
integ=a0+a1*(theta1*s+theta2*u)+a2*(Sigma+(theta1*s+theta2*u)^2);
r=Q*s^2+R*u^2;
%%%%%%%%%%%%%%Bellman 
ob=r+gamma*integ;
end 

function optu=optCtrl(s,a)
%%%%%%%%%%%%parameters
%Sigma=0.01;
R=10;
%Q=1;
gamma=0.5;
theta1=-3;
theta2=2;
%%%%%%%%assign a
a0=a(1); a1=a(2); a2=a(3);
%%%%%%%%%%%
num=gamma*(a1*theta1+2*a2*theta1*theta2*s);
den=2*(R+gamma*a2*theta2^2);
%%%%%%%%%%%%
optu=-1*num/den;
end

