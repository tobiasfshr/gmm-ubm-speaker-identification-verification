function new_array = convert_2d(old_array)
    % convert 3D array into 2D...
    % Input: old_array  : 3d array of shape 1xMxN
    % Output: new_array : 2d array of shape MxN
    new_array = reshape(old_array, [size(old_array, 2), size(old_array, 3)]);
end