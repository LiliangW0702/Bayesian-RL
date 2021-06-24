clear all;
close all; 
clc;



a_limit=1000;
s_limit=20000; %sampling limit of state
N=2000;

%%%%%%%%%%%%%%
a_pre=a_limit*rand(3,1); %initialize the function weight a
% Sigma=0.01;
% R=10;
% Q=1;
% gamma=0.5;
% theta1=-3;
% theta2=2;
% [P,~,~]=dare(theta1*sqrt(gamma),theta2,Q,R/gamma);
% a0=gamma/(1-gamma)*P*Sigma; %true a0
% a2=P; %true a2 
% a1=0;
% a_pre=[a0;a1;a2];

A=[];
b=[];
x0=10;
V=zeros(N,1);
x=zeros(N,1);
tolerence=0.000001;
change=1000;
j=0;
Er=[];
V_test=[];
k=1;
s_test=4.0;

%%%%%%%%%%%%test
computeOptCtrl_test();%test optimal control computation function
computeV_test();%test the function which computes the RHS of Bellman equation
value_function_test();
error_function_test();

%%%%%%%%randomly sample N states
for i=1:N
    s=(-1+2*rand)*s_limit;
    x(i)=s;
end
%%%%%%%%%%%%
while j<10 %change>tolerence
    for i=1:N
        s=x(i);
        u_opt=optCtrl(s,a_pre);
        V(i)=computeV(s,u_opt,a_pre); %compute RHS of Bellman eqaution for each sample point
    end

fun2=@(a)error(V,x,a);
[a_opt,err]=fmincon(fun2,a_pre,A,b);
u_opt_test=optCtrl(s_test,a_opt);
%V_test(k)=computeV(s_test,u_opt_test,a_opt);
V_test(k)=value_function(s_test,a_pre);
change=norm(a_opt-a_pre);%/norm(a_pre);
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

function error_function_test()
s_limit=200;
s=(-1+2*rand)*s_limit; %tets state
%%%%%%%%%%%%parameters
Sigma=0.01;
R=10;
Q=1;
gamma=0.5;
theta1=-3;
theta2=2;
[P,~,~]=dare(theta1*sqrt(gamma),theta2,Q,R/gamma);
a0=gamma/(1-gamma)*P*Sigma; %true a0
a2=P; %true a2 
a1=0;
a=[a0;a1;a2];
uopt=optCtrl(s,a);
V=computeV(s,uopt,a);
result=error(V,s,a);
assert ( (result)^2< 1e-7,'Wrong error function');
end



function err=error(V,x,a)
N=length(V);
for i=1:N
    err_v=zeros(N,1);
    V_LHS=value_function(x(i),a);%Left hand side of Bellman Equation
    err_v(i)=V_LHS-V(i);
    err=norm(err_v);
end
end

function value_function_test()
s=2;
a=[-1;3;5];
value_tr=25;
result=value_function(s,a);
assert ( (result-value_tr)^2< 1e-7,'Wrong ComputeV function');
end


function V=value_function(s,a)
%%%%%%%%assign a
a0=a(1); a1=a(2); a2=a(3);
V=a0+a1*s+a2*s^2;
end

function computeV_test()
s_limit=200;
s=(-1+2*rand)*s_limit; %tets state
%%%%%%%%%%%%parameters
Sigma=0.01;
R=10;
Q=1;
gamma=0.5;
theta1=-3;
theta2=2;
[P,~,~]=dare(theta1*sqrt(gamma),theta2,Q,R/gamma);
V_tr=P*s^2+gamma/(1-gamma)*P*Sigma;%true value function for optimal control 

K=gamma*theta2*P*theta1/(R+gamma*theta2^2*P);  %true optimal control gain
u_tr=-1*K*s; %true optimal control
a0=gamma/(1-gamma)*P*Sigma; %true a0
a2=P; %true a2 
a1=0;
a=[a0;a1;a2];
result=computeV(s,u_tr,a);
disp(V_tr);
disp(result);
assert ( (result-V_tr)^2< 1e-7,'Wrong ComputeV function');
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

function computeOptCtrl_test()
s=1; %tets state
%%%%%%%%%%%%parameters
Sigma=0.01;
R=10;
Q=1;
gamma=0.5;
theta1=-3;
theta2=2;
[P,~,~]=dare(theta1*sqrt(gamma),theta2,Q,R/gamma);
K=gamma*theta2*P*theta1/(R+gamma*theta2^2*P);  %true optimal control gain
u_tr=-1*K*s; %true optimal control
a0=gamma*(1-gamma)*P*Sigma; %true a0
a2=P; %true a2 
a1=0;
a=[a0;a1;a2];
result=optCtrl(s,a);%computed optimal control
disp(u_tr);
disp(result);
disp(result-u_tr);
assert ( (result-u_tr)^2< 1e-7,'Wrong optCtrl function');
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
