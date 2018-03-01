function logLikelihood = getLogLikelihood(means, weights, covariances, X)
	% Log Likelihood estimation
	%
	% INPUT:
	% means          : Mean for each Gaussian KxD
	% weights        : Weight vector 1xK for K Gaussians
	% covariances    : Covariance matrices for each gaussian DxDxK
	% X              : Input data NxD
	%%% where N is number of data points
	%%% D is the dimension of the data points
	%%% K is number of gaussians
	%
	% OUTPUT:
	% logLikelihood  : log-likelihood

	% get dimensions of an input data
	[N, D] = size(X);

	% get number of gaussians
	K = length(weights);

	logLikelihood = 0;
	for i = 1:N % For each of the data points
		% probability p
		p = 0;
		for j = 1:K  % For each of the mixture components
			meansDiff = X(i,:) - means(j,:);
			covariance = covariances(:,:,j);
			norm = 1 / (((2*pi)^(D/2)) * sqrt(det(covariance)));
			p = p + weights(1,j) * norm * exp(-0.5 * ((meansDiff / covariance) * meansDiff') );
		end
		logLikelihood = logLikelihood + log(p);
    end
end

