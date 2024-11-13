%code to find reversibility on good outcome patients
%% this script has delay changed to 15 original was 1 
clear all
clc
Tau=18; 
TR=0.05; 
FS=200;
fnq=FS/2;                 % Nyquist frequency
flp = 4;                    % lowpass frequency of filter (Hz)
fhi = 8;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency 
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter
findings=table(); %CREATES AN EMPTY TABLE
final_findings=table();
%% MAKING THE EPOCHS CONTIGUIOUS
folder_path = '/Users/samuelborromeo/Documents/MATLAB/separate good';
A= dir(fullfile(folder_path, '*.set'));

if isempty(A)
    disp('No .set files found in the specified directory.');
    return;
end

cd(folder_path);

for ii=1:size(A,1)
    ii
    load('-mat',A(ii).name) %LOADS THE FIRST SET FILE
    disp(A(ii).name)
    %FOR EACH SET FILE WE GET A DATA VARIABLE A 3D MATRIX
    %WHICH HAS DIMENSIONS 24x200x 1150 FOR EXAMPLE
    %CH x TIME X SAMPLES
    
    
    
    %CONCATENATES THE EEG EPOCHS IE CONVERTS FROM 3D TO 2D NOW DIMENSION SHOULD BE
    %CH x SAMPLES
    data_2d=[]; %DATA 2D HOLDS THE EPOCHS AFTER THEY ARE MADE CONTIGUIOUS
    
    for i=1:size(data,3)
       
        epoch = data(:,:,i);
        epoch = detrend(epoch - nanmean(epoch));
        epoch = filtfilt(bfilt, afilt, epoch); %filter band
        data_2d = [data_2d epoch];
    end
    
    
    %% ROW WISE DETRENDING BAND PASS FILTERING (1 TO 4HZ) OF THE DATA_2D VARIABLE
    for row=1:size(data_2d,1)
        data_2d(row,:)=detrend(data_2d(row,:)-nanmean(data_2d(row,:)));
        %data_2d(row,:)=filtfilt(data_2d(row,:),[1 4],200); %FILTERED BETWEEN 1 TO 4 HZ, (DELTA BAND) WITH SAMPLE FREQ OF 200

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
    


end






