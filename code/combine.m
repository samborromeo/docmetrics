clc; 
clear;

DATA=[]; %holds all the sessions together
lengths_of_eachfile=[]; % we can store each record length after making continuious
A= dir('*.set');


for ii=1:size(A,1)
    ii;
    load('-mat',A(ii).name) %LOADS THE FIRST SET FILE
cont_eeg=makecont(data);
DATA=[DATA cont_eeg ]; %adds next session data to  DATA
l=length(cont_eeg);%length of current session

lengths_of_eachfile=[lengths_of_eachfile l];

end

totallength=sum(lengths_of_eachfile);
if(totallength==length(DATA))
    fprintf("data addition successful")
end


data=DATA;
save('alldata.mat','data');

%%%%%%%%%%%%%% IRREVERSIBILITY ANNALYSIS %%%%%%%%%%%%%%%%%%%

Tau=15; 
TR=0.05; 
FS=200;
fnq=FS/2;                 % Nyquist frequency
flp = 1;                    % lowpass frequency of filter (Hz)
fhi = 45;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency 
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter
findings=table(); %CREATES AN EMPTY TABLE
final_findings=table();
%% MAKING THE EPOCHS CONTIGUIOUS

% Determine the final dimensions
num_epochs = size(data, 2);
num_samples_per_epoch = size(data, 1);

% Preallocate data_2d
data_2d = zeros(num_samples_per_epoch, num_epochs);


for i=1:size(data,2) 
    epoch = data(:,i);
    epoch = filtfilt(bfilt, afilt, epoch); %filter band
    data_2d(:, i) = epoch;
end

for row=1:size(data_2d,1)
    data_2d(row,:)=detrend(data_2d(row,:)-nanmean(data_2d(row,:)));
end

Tm=size(data_2d,2); %LENGTH OF TIME SERIES AFTER CONATENATION OF EPOCHS
%% DROPPING THE LAST 3 CHANNELS OR ROWS (2EOG +1 ECG) 22nd 23rd and 24th
FCtf=corr(data_2d(1:21,1:Tm-Tau)',data_2d(1:21,1+Tau:Tm)'); % has dimension 21 X 21 
FCtr=corr(data_2d(1:21,Tm:-1:Tau+1)',data_2d(1:21,Tm-Tau:-1:1)'); % has dimension 21 X 21
    
%Irr=abs(FCtf-FCtr); % we are not using this
    
% MUTUAL INFORMATION
Itauf=-0.5*log(1-FCtf.*FCtf); % has dimension 21 X 21
Itaur=-0.5*log(1-FCtr.*FCtr); % has dimension 21X 21
Reference(ii,:)=((Itauf(:)-Itaur(:)).^2)'; % this is the irrversibility score has dimension 1X 441