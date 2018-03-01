function p = parameters()
	p.epsilon = 0.0001; % regularization
	p.K = 64; % number of desired clusters
	p.n_iter = 5; % number of iterations
	p.core_set_indices = [167, 172, 229, 233, 238, 239, 241, 242, 243, 244]; % Core set of speakers
    p.yoho_path = 'C:\Users\Tobias\Desktop\YOHO\';
    p.background_set_indices = [124,146,147,140,110,115,120,138,133,104,105,107,108,109,116,122,142,143,101,103,106,111,112,113,114,117,118,119,121,125,126,127,130,131,132,134,135,137,141,148,150,139,128,136,102,144,145];
    p.impostor_set_indices = [188,198,181,192,152,159,163,164,156,157,158,161];
    p.r = 9; %relevance factor for MAP adaptation
end

