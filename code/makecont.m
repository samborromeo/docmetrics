%% make epochs contiguious 
function cont_eeg=makecont(data)

cont_eeg=[]; %DATA 2D HOLDS THE EPOCHS AFTER THEY ARE MADE CONTIGUIOUS
    for i=1:size(data,3)
        cont_eeg=[cont_eeg data(:,:,i)];
    end
    
end

