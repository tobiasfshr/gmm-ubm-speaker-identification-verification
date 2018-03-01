function [weights, means, covariances, logLikelihood] = MStep(gamma, X)
	% Maximization step of the EM Algorithm
	%
	% INPUT:
	% gamma          : NxK matrix of responsibilities for N datapoints and K gaussians.
	% X              : Input data (NxD matrix for N datapoints of dimension D).
	%
	%%% N is number of data points
	%%% D is the dimension of the data points
	%%% K is number of gaussians
	%
	% OUTPUT:
	% logLikelihood  : Log-likelihood (a scalar).
	% means          : Mean for each gaussian (KxD).
	% weights        : Vector of weights of each gaussian (1xK).
	% covariances    : Covariance matrices for each component(DxDxK).

	% Get the sizes
	[nTrainingSamples, dim] = size(X);
	K = size(gamma,2);

	noloops = 0;
	if noloops

		% update the weights
		N_hat = sum(gamma, 1);
		weights = N_hat/nTrainingSamples;

		% update means
		means = bsxfun(@rdivide, gamma'*X, N_hat');

		% update covariances
		covariances = zeros(dim, dim, K);
		for j = 1:K
			debiased = X - repmat(means(j, :), nTrainingSamples, 1);
			covariances(:, :, j) = (bsxfun(@times, debiased, gamma(:, j))' * debiased) / N_hat(j);
		end

	else

		% Or use for loops

		% Create matrices
		means = zeros(K, dim);
		covariances = zeros(dim, dim, K);

		% Compute the weights
		Nk = sum(gamma, 1);
		weights = Nk./nTrainingSamples;

		for i=1:K
			auxMean = zeros(1, dim);
			for j=1:nTrainingSamples
				auxMean = auxMean + gamma(j, i) .* X(j, :);
			end

			means(i, :) = auxMean./Nk(i);
		end

		for i=1:K
			auxSigma = zeros(dim, dim);
			for j=1:nTrainingSamples
				meansDiff = X(j, :) - means(i, :);
				auxSigma = auxSigma + gamma(j, i).*((meansDiff)' * meansDiff);
			end

			covariances(:,:,i) = auxSigma ./ Nk(i);
		end

	end

	% compute loglikelihood
	logLikelihood = getLogLikelihood(means, weights, covariances, X);



end

