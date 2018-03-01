function chunked_data = createChunks(data, numChunks)
% get LogLikelihood Ratio for speaker verification
% Input:  data            : 2D array of data points (MxN)
%         numChunks       : number of chunks to create (K)
% Output: chunked_data    : 3d array with data chunks (Kx(M/K)xN)
    count = length(data)-mod(size(data,1), numChunks); %number of elements must be dividable by numChunks to reshape
    tmp_data = data(1:count, :);
    chunked_data = reshape(tmp_data, [uint16(length(tmp_data)/numChunks), numChunks, 12]);
end