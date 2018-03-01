function plotModes(means, covMats, X)

	scatter(X(:, 1), X(:, 2));
	hold on;
	M = size(means, 2);

	for i=1:M
		plotGaussian(means(:, i), covMats(:, :, i));
	end

	hold off;
	drawnow;

end

