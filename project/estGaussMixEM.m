function [weights, means, covariances] = estGaussMixEM(data, K, n_iters, epsilon)
	% EM algorithm for estimation gaussian mixture mode
	%
	% INPUT:
	% data           : input data, N observations, D dimensional
	% K              : number of mixture components (modes)
	%
	% OUTPUT:
	% weights        : mixture weights
	% means          : means of gaussians
	% covariances    : covariances matrices of gaussians

	n_dim = size(data, 2);

	% initialize weights and covariances
	weights = ones(1, K)./K;
	covariances = zeros(n_dim, n_dim, K);

	% Use k-means for initializing the EM-Algorithm.
	% cluster_idx: cluster indices
	% means: cluster centers
	% sumd: sum of distances in cluster
	[cluster_idx, means, sumd] = kmeans(data, K, 'replicate', 10);

	% Create initial covariance matrices
	for j = 1:K
		sumd(j, 1) = sumd(j, 1) ./ sum(cluster_idx == j);
	end

	for j = 1:K
		covariances(:, :, j) = eye(n_dim) .* sumd(j, 1);
	end

	% Run EM
	for iter = 1:n_iters

		[oldLogLi, gamma] = EStep(means, covariances, weights, data);
		[weights, means, covariances, newLogli] = MStep(gamma, data);

		% regularize covariance matrix
		for j = 1:K
			covariances(:, :, j) = regularize_cov(covariances(:, :, j), epsilon);
		end

		% termination criterion
		if abs(oldLogLi - newLogli) < 1
			break;
		end
	end

end

