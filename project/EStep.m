function [logLikelihood, gamma] = EStep(means, covariances, weights, X)
	% Expectation step of the EM Algorithm
	%
	% INPUT:
	% means          : Mean for each Gaussian KxD
	% weights        : Weight vector 1xK for K Gaussians
	% covariances    : Covariance matrices for each gaussian DxDxK
	% X              : Input data NxD
	%
	%%% N is number of data points
	%%% D is the dimension of the data points
	%%% K is number of gaussians
	%
	% OUTPUT:
	% logLikelihood  : Log-likelihood (a scalar).
	% gamma          : NxK matrix of responsibilities for N datapoints and K gaussians.

	logLikelihood = getLogLikelihood(means, weights, covariances, X);

		[nTrainingSamples, dim] = size(X);
		K = size(weights, 2);

		gamma = zeros(nTrainingSamples, K);
		for i = 1:nTrainingSamples
			for j = 1:K
				meansDiff = X(i, :) - means(j, :);
				covariance = covariances(:,:,j);
				norm = 1 / (((2*pi)^(dim/2)) * sqrt(det(covariances(:, :, j))));

				gamma(i, j) = weights(1, j)*norm*exp(-0.5*((meansDiff / covariance)*meansDiff'));
			end
			gamma(i, :) = gamma(i, :)./sum(gamma(i, :));
        end
end

