function data = getYohoData(path, speakerIdx, mode)
	% front-end of verfication system
	%
	% INPUT:
	% path          : Mean for each Gaussian KxD
    % speakerIdx    : index of speaker
    % mode          : 'ENROLL' - get training data
    %                 'VERIFY' - get test data
	%
	% OUTPUT:
	% data          :  matrix of all MFCC feature vectors of speaker
    data = [];
    folderpath = strcat(path, mode, '\', num2str(speakerIdx), '\');
    cd(folderpath);
    folderlist = ls;
    for i = 1:length(folderlist)
        filepath = strcat(folderpath, num2str(folderlist(i, :)));
        cd(filepath);
        filelist = ls('*.wav');
        
        for k = 1:size(filelist, 1)
            [s, fs] = readsph(filelist(k, :), 'wt');
            %extract speech with VAD
            [vs,zo] = vadsohn(s,fs);
            speech_s = [];
            for i = 1:length(vs)
                if (vs(i) == 1)
                    speech_s = [speech_s; s(i)];
                end
            end    
            %convert speech into MFCC vector and add to data
            [c, tc] = melcepst(speech_s, fs);
            data = [data; c];
        end  
    end  
    
end
