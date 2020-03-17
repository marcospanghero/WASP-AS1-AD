Ts=0.01;
Tmax=60;
t=0:Ts:Tmax;
N=length(t);

% Initialize state vector x
x=zeros(10,1);
x(end)=1;

% Try out different biases
accbias = [0;0;0];
gyrobias = [0.01*pi/180;0;0]
u = [gravity(59,0) + accbias ; gyrobias];

% simulate IMU standalone navigation system
pos=zeros(3,N);
for n=2:N
   x = Nav_eq(x,u,Ts);
   pos(:,n) = x(1:3);
end
figure(1)
clf
plot(t,pos')
grid on
ylabel('Position error [m]')
xlabel('Time [s]')
legend
print("img/error-growth-position",'-dpng');


figure(2)
clf
tpos = pos'
plot3(tpos(:,1),tpos(:,2),tpos(:,3))
grid on
legend
print("img/error-growth-position-3d",'-dpng');


figure(3)
loglog(t,pos')
pos2=9.82*gyrobias(1)*t.^3/6;  % theoretical 
pos3=9.82*(gyrobias(1))^2*t.^4/24; % theoretical
hold on
loglog(t,pos2,'k--')
loglog(t,pos3,'k--')
print("img/error-growth-bias",'-dpng');



