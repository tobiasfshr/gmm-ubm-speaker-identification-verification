function regularized_cov = regularize_cov(covariance, epsilon)
	% regularize a covariance matrix, by enforcing a minimum
	% value on its singular values. Explanation see exercise sheet.
	%
	% INPUT:
	%  covariance: matrix
	%  epsilon:    minimum value for singular values
	%
	% OUTPUT:
	% regularized_cov: reconstructed matrix

	regularized_cov = covariance + epsilon * eye(size(covariance));

	% makes sure matrix is symmetric upto 1e-15 desimal
	regularized_cov = (regularized_cov + regularized_cov')/2;

end
