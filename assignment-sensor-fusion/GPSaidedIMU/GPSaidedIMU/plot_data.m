function plot_data(in_data,out_data,holdon,settings)

if nargin < 3; holdon = 'False'; end

h=zeros(1,2);
fig = figure(1);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact')
if ~holdon clf; end
slowINS_x = decimate(out_data.x_h(2,:),100);
slowINS_y = decimate(out_data.x_h(1,:),100);
fprintf("Former Len %d\n",length(out_data.x_h(2,:)));
fprintf("Decimate Len %d\n",length(slowINS_x));

nexttile
hold on
h(1)=plot(in_data.GNSS.pos_ned(2,:),in_data.GNSS.pos_ned(1,:),'b.');
h(2)=plot(slowINS_x,slowINS_y,'r.');
title('Position error')
ylabel('North [m]')
xlabel('East [m]')
legend(h,'GNSS position estimate','GNSS aided INS trajectory','Start point')
axis equal
grid on

xerr = slowINS_x - in_data.GNSS.pos_ned(2,:); 
yerr = slowINS_y - in_data.GNSS.pos_ned(1,:);
error = sqrt(xerr.^2 + yerr.^2);
positiodec_RMS = sqrt(mean(sqrt(xerr.^2 + yerr.^2)))

p = zeros(1,2);
nexttile
hold on
p(1) = semilogy(error);
p(2) = yline(positiodec_RMS);
title('Position error')
xlabel('Sample')
ylabel('Norm Error in [m]')
legend(p,'Norm Error', 'RMS')
grid on

p = zeros(1,2);
nexttile
hold on
p(1) = plot(xerr);
p(2) = plot(yerr);
title('Position error x,y')
xlabel('Sample')
ylabel('Error in m')
legend(p, 'xError', 'yError')
grid on
print("img/" + settings.savepath +"error-dec",'-dpng');



h=zeros(1,4);
fig = figure(2);
if ~holdon clf; end
plot(in_data.GNSS.pos_ned(2,:),in_data.GNSS.pos_ned(1,:),'b-');
hold on
h(1)=plot(in_data.GNSS.pos_ned(2,:),in_data.GNSS.pos_ned(1,:),'b.');
h(2)=plot(out_data.x_h(2,:),out_data.x_h(1,:),'r');
h(3)=plot(in_data.GNSS.pos_ned(2,1),in_data.GNSS.pos_ned(1,1),'ks');
h(4)=plot(in_data.GNSS.pos_ned(2,settings.outagestart),in_data.GNSS.pos_ned(1,settings.outagestart),'k^');
%h(5)=plot(in_data.GNSS.pos_ned(2,settings.outagestop),in_data.GNSS.pos_ned(1,settings.outagestop),'k^');

title('Trajectory')
ylabel('North [m]')
xlabel('East [m]')
legend(h,'GNSS position estimate','GNSS aided INS trajectory','Start point')
axis equal
grid on
print("img/" + settings.savepath +"error-trajectory",'-dpng');

h=zeros(1,3);
fig = figure(3);
if ~holdon clf; end
h(1)=plot(in_data.GNSS.t,-in_data.GNSS.pos_ned(3,:),'b.');
hold on
h(2)=plot(in_data.IMU.t,-out_data.x_h(3,:),'r');
h(3)=plot(in_data.IMU.t,3*sqrt(out_data.diag_P(3,:))-out_data.x_h(3,:),'k--');
plot(in_data.IMU.t,-3*sqrt(out_data.diag_P(3,:))-out_data.x_h(3,:),'k--')
title('Height versus time')
ylabel('Height [m]')
xlabel('Time [s]')
grid on
legend(h,'GNSS estimate','GNSS aided INS estimate','3\sigma bound')
print("img/" + settings.savepath +"error-gnss-height",'-dpng');


h=zeros(1,2);
fig = figure(4);
if ~holdon clf; end
h(1)=plot(in_data.IMU.t,sqrt(sum(out_data.x_h(4:6,:).^2)),'r');
hold on
sigma=sqrt(sum(out_data.diag_P(4:6,:)));
speed=sqrt(sum(out_data.x_h(4:6,:).^2));
h(2)=plot(in_data.IMU.t,3*sigma+speed,'k--');
plot(in_data.IMU.t,-3*sigma+speed,'k--')
title('Speed versus time')
ylabel('Speed [m/s]')
xlabel('Time [s]')
grid on
legend(h,'GNSS aided INS estimate','3\sigma bound')

N=length(in_data.IMU.t);
attitude=zeros(3,N);

for n=1:N
    Rb2t=q2dcm(out_data.x_h(7:10,n));
    
    %Get the roll, pitch and yaw
    % roll
    attitude(1,n)=atan2(Rb2t(3,2),Rb2t(3,3))*180/pi;
    
    % pitch
    attitude(2,n)=-atan(Rb2t(3,1)/sqrt(1-Rb2t(3,1)^2))*180/pi;
    
    %yaw
    attitude(3,n)=atan2(Rb2t(2,1),Rb2t(1,1))*180/pi;
end

ylabels={'Roll [deg]','Pitch [deg]','Yaw [deg]'};
print("img/" + settings.savepath +"error-gnss-speed",'-dpng');


fig = figure(5);
if ~holdon clf; end
h=zeros(1,2);
for k=1:3
    subplot(3,1,k)
    h(1)=plot(in_data.IMU.t,attitude(k,:),'r');
    hold on
    plot(in_data.IMU.t,3*180/pi*sqrt(out_data.diag_P(6+k,:))+attitude(k,:),'k--')
    h(2)=plot(in_data.IMU.t,-3*180/pi*sqrt(out_data.diag_P(6+k,:))+attitude(k,:),'k--');
    grid on
    ylabel(ylabels{k})
    if k==1
        title('Attitude versus time')
    end
end
xlabel('Time [s]')
legend(h,'GNSS aided INS estimate','3\sigma bound')
print("img/" + settings.savepath +"error-gnss-attitude",'-dpng');


fig = figure(6);
if ~holdon clf; end
ylabels={'X-axis bias [m/s^2]','Y-axis bias [m/s^2]','Z-axis bias [m/s^2]'};
h=zeros(1,2);
for k=1:3
    subplot(3,1,k)
    h(1)=plot(in_data.IMU.t,out_data.delta_u_h(k,:),'r');
    hold on
    h(2)=plot(in_data.IMU.t,3*sqrt(out_data.diag_P(9+k,:))+out_data.delta_u_h(k,:),'k--');
    plot(in_data.IMU.t,-3*sqrt(out_data.diag_P(9+k,:))+out_data.delta_u_h(k,:),'k--')
    grid on
    ylabel(ylabels{k})
    if k==1
        title('Accelerometer bias estimate versus time')
    end
end
xlabel('Time [s]')
legend(h,'GNSS aided INS estimate','3\sigma bound')
print("img/" + settings.savepath +"error-error-acc-time",'-dpng');


fig = figure(7);
if ~holdon clf; end
ylabels={'X-axis bias [deg/s]','Y-axis bias [deg/s]','Z-axis bias [deg/s]'};
h=zeros(1,2);
for k=1:3
    subplot(3,1,k)
    h(1)=plot(in_data.IMU.t,180/pi*out_data.delta_u_h(3+k,:),'r');
    hold on
    h(2)=plot(in_data.IMU.t,3*180/pi*sqrt(out_data.diag_P(12+k,:))+180/pi*out_data.delta_u_h(3+k,:),'k--');
    plot(in_data.IMU.t,-3*180/pi*sqrt(out_data.diag_P(12+k,:))+180/pi*out_data.delta_u_h(3+k,:),'k--')
    grid on
    ylabel(ylabels{k})
    if k==1
        title('Gyroscope bias estimate versus time')
    end
end
xlabel('Time [s]')
legend(h,'GNSS aided INS estimate','3\sigma bound')
print("img/" + settings.savepath +"error-error-gyro-time",'-dpng');


fig = figure(8);
if ~holdon clf; end
xest = out_data.x_h(2,:);
yest = out_data.x_h(1,:);
xgps = interp1(in_data.GNSS.t,in_data.GNSS.pos_ned(2,:),in_data.IMU.t,'linear','extrap')';
ygps = interp1(in_data.GNSS.t,in_data.GNSS.pos_ned(1,:),in_data.IMU.t,'linear','extrap')';
xerr = xest - xgps; 
yerr = yest - ygps;
plot(in_data.IMU.t,xerr)
grid on
hold on
plot(in_data.IMU.t,yerr)
xlabel('time [s]')
ylabel('position difference [m]')
legend('x', 'y')
print("img/" + settings.savepath +"ierror-error-position-diff",'-dpng');

positionerr_RMS = sqrt(mean(xerr.^2+yerr.^2))
fileID = fopen("img/" + settings.savepath +"err.txt",'w');
fprintf(fileID,'positionerr_RMS = %f',positionerr_RMS);
fclose(fileID);

end

