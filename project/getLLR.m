function LLR = getLLR(gmm_ubm, ubm, current_utterance)
% get LogLikelihood Ratio for speaker verification
% Input:  gmm_ubm           : speaker model
%         ubm               : universal background model
%         current_utterance : data for LRR computation
% Output: LLR               : LLR of speaker model and ubm
    likelihood_gmm_ubm = getLogLikelihood(gmm_ubm.mu, gmm_ubm.w, gmm_ubm.sigma, current_utterance);
    likelihood_ubm = getLogLikelihood(ubm.mu, ubm.w, ubm.sigma, current_utterance);
    LLR = (likelihood_gmm_ubm - likelihood_ubm)/length(current_utterance);
end