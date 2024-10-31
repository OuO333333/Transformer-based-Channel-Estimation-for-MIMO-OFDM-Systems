%%  3GPP TR 38.901 release 15
% generate candidates for channel statistics

clear
Nc=64;
L=3; % number of path
Nt=32;
Nr=16;
num_para=6;
num_sta=50;
channel_statistic=zeros(num_para,L,num_sta);
% 存储所有最大时延值
max_delays = zeros(1, num_sta);
path_gains = zeros(1, num_sta);
path_gains_dB = zeros(1, num_sta);
Angles_AoA = zeros(1, num_sta);

for m=1:num_sta
LOSangle=60;
philosAoA=LOSangle;
philosAoD=LOSangle;
thetalosZoA=LOSangle;
thetalosZoD=LOSangle;
cdl = nrCDLChannel;
cdl.CarrierFrequency=28e9;
cdl.TransmitAntennaArray.Size = [Nt 1 1 1 1];
cdl.ReceiveAntennaArray.Size = [Nr 1 1 1 1];
fc=cdl.CarrierFrequency/1e9;
cdl.MaximumDopplerShift = 0;
cdl.ChannelFiltering=false;    
cdl.DelayProfile='Custom';        
lgDSmean=-0.24*log10(1+ fc) - 6.83;   %UMi - Street Canyon  NLOS
lgDSstanvar=0.16*log10(1+ fc) + 0.28;
DS=10^(normrnd(lgDSmean,lgDSstanvar));  % DS
r_tau=2.1;
tau=-r_tau*DS*log(rand(1,L));
cdl.PathDelays=sort(tau-min(tau));   % path delays 
save_PD=cdl.PathDelays;
Power=exp(-(r_tau-1)/(r_tau*DS)*cdl.PathDelays).*10.^(-normrnd(0,3,1,L)/10);
P=Power/sum(Power);
cdl.AveragePathGains=10*log10(P);  % average path gains
save_P=cdl.AveragePathGains;
lgASAmean=-0.08*log10(1+ fc) + 1.81;   %UMi - Street Canyon  NLOS
lgASAstanvar=0.05*log10(1+ fc) + 0.3;
ASA=10^(normrnd(lgASAmean,lgASAstanvar));  % ASA
Cphi=0.779; % N=4
Phi=2*ASA/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoA=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASA/7,1,L)+philosAoA;  % AoA
save_AoA=cdl.AnglesAoA;
lgASDmean=-0.23*log10(1+ fc) + 1.53;   %UMi - Street Canyon  NLOS
lgASDstanvar=0.11*log10(1+ fc) + 0.33;
ASD=10^(normrnd(lgASDmean,lgASDstanvar));  % ASD
Cphi=0.779; % N=4
Phi=2*ASD/1.4*sqrt(-log(P/max(P)))/Cphi;
cdl.AnglesAoD=(2*randi([0,1],1,L)-1).*Phi+normrnd(0,ASD/7,1,L)+philosAoD;  % AoD
save_AoD=cdl.AnglesAoD;
lgZSAmean=-0.04*log10(1+ fc) + 0.92;   %UMi - Street Canyon  NLOS
lgZSAstanvar=-0.07*log10(1+ fc) + 0.41;
ZSA=10^(normrnd(lgZSAmean,lgZSAstanvar));  % ZSA
Ctheta=0.889; % N=8
Theta=-ZSA*log(P/max(P))/Ctheta;
cdl.AnglesZoA=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSA/7,1,L)+thetalosZoA;  % ZoA
save_ZoA=cdl.AnglesZoA;
d2D=50;
hUT=0;
hBS=10;
lgZSDmean=max(-0.5, -3.1*(d2D/1000)+ 0.01*max(hUT-hBS,0) +0.2);   %UMi - Street Canyon  NLOS
lgZSDstanvar=0.35;
ZSD=10^(normrnd(lgZSDmean,lgZSDstanvar));  % ZSD
Ctheta=0.889; % N=8
Theta=-ZSD*log(P/max(P))/Ctheta;
cdl.AnglesZoD=(2*randi([0,1],1,L)-1).*Theta+normrnd(0,ZSD/7,1,L)-10^(-1.5*log10(max(10, d2D))+3.3)+thetalosZoD;  % ZoD
save_ZoD=cdl.AnglesZoD;
sta_mtx=[save_PD;save_P;save_AoA;save_AoD;save_ZoA;save_ZoD];
channel_statistic(:,:,m)=sta_mtx;

% 计算最大时延并存储
    max_delays(m) = max(cdl.PathDelays);
    % 计算路径增益并存储
    path_gains(m) = max(P); % 使用原始的线性比例
    path_gains_dB(m) = max(cdl.AveragePathGains); % 使用dB值
    Angles_AoA(m) = max(cdl.AnglesAoA);
    disp(['Maximum Delay (τmax) for m=', num2str(m), ': ', num2str(max_delays(m)), ' seconds']);
    disp(['Maximum Path Gain (βl) for m=', num2str(m), ': ', num2str(path_gains_dB(m)), ' dB']);
    disp(['Linear Path Gain (βl) for m=', num2str(m), ': ', num2str(path_gains(m))]);
    disp(['AoA for m=', num2str(m), ': ', num2str(Angles_AoA(m)), ' degrees']);
end
save ch_sta_mtx channel_statistic

% 计算最大时延和路径增益的统计特性
mean_delay = mean(max_delays);
std_delay = std(max_delays);
min_delay = min(max_delays);
max_delay = max(max_delays);

mean_gain = mean(path_gains);
std_gain = std(path_gains);
min_gain = min(path_gains);
max_gain = max(path_gains);

mean_gain_dB = mean(path_gains_dB);
std_gain_dB = std(path_gains_dB);
min_gain_dB = min(path_gains_dB);
max_gain_dB = max(path_gains_dB);

mean_AoA = mean(Angles_AoA);
std_AoA = std(Angles_AoA);
min_AoA = min(Angles_AoA);
max_AoA = max(Angles_AoA);


% 输出统计特性
disp(['Mean Maximum Delay: ', num2str(mean_delay), ' seconds']);
disp(['Standard Deviation of Maximum Delay: ', num2str(std_delay), ' seconds']);
disp(['Minimum Maximum Delay: ', num2str(min_delay), ' seconds']);
disp(['Overall Maximum Delay: ', num2str(max_delay), ' seconds']);

disp(['Mean Path Gain: ', num2str(mean_gain)]);
disp(['Standard Deviation of Path Gain: ', num2str(std_gain)]);
disp(['Minimum Path Gain: ', num2str(min_gain)]);
disp(['Overall Maximum Path Gain: ', num2str(max_gain)]);

disp(['Mean Path Gain(dB): ', num2str(mean_gain_dB), ' dB']);
disp(['Standard Deviation of Path Gain(dB): ', num2str(std_gain_dB), ' dB']);
disp(['Minimum Path Gain(dB): ', num2str(min_gain_dB), ' dB']);
disp(['Overall Maximum Path Gain(dB): ', num2str(max_gain_dB), ' dB']);

disp(['Mean AoA: ', num2str(mean_AoA), ' degrees']);
disp(['Standard Deviation of AoA: ', num2str(std_AoA), ' degrees']);
disp(['Minimum AoA: ', num2str(min_AoA), ' degrees']);
disp(['Overall Maximum AoA: ', num2str(max_AoA), ' degrees']);