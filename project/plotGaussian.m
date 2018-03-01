function plotGaussian(mu, sigma)

	[dimension, nComponents] = size(mu);

	if(dimension == 1)
		if (nComponents == 1 & size(sigma) == [1, 1])
			y = [];
			for x = mu-4*sigma:.1:mu+4*sigma
				y = [y, [x, exp(-(x-mu)'*(x-mu)/2/sigma^2)/sqrt(2*pi*sigma)]'];
			end
			plot(y(1, :), y(2, :));
		else
			sprintf('ERROR: size mismatch in mu or sigma\n');
		end
	elseif(dimension == 2)
		if(nComponents == 1 & size(sigma) == [2, 2])
			n = 36;
			phi = (0:1:n-1)/(n-1)*2*pi;
			epoints = sqrtm(sigma) * [cos(phi); sin(phi)] + mu*ones(1, n);
			plot(epoints(1, :), epoints(2, :), 'r');
		else
			sprintf('ERROR: size mismatch in mu or sigma\n');
		end
	end

end

