import numpy as np

def convolve_2d(image : np.ndarray, kernel: np.ndarray, stride=1, keep_size=False, padding_mode='constant'):
    """
    Applys a convultion kernel to an image.
    Parameters:
        image (np.ndarray): Input image data.
        kernel (np.ndarray): The mask to be convolved across the image.
        stride (np.ndarray): Amount in pixels by which the mask should step in both x and y across the image.
        keep_size (np.ndarray): If True the image is padded before convolution to maintain the size shape after the mask is applied.
        padding_mode (np.ndarray): Passed to numpy's pad function.
    Returns:
        image (np.ndarray): The convolved image.
    """

    if stride > 1 and keep_size:
        print("Warn: Stride > 1 and keep_size cannot both be true. Setting to false")
        keep_size = False
    
    input_image_shape = image.shape
    kernel_shape = kernel.shape

    output_shape_x = int(((input_image_shape[0] - kernel_shape[0]) / stride) + 1)
    output_shape_y = int(((input_image_shape[1] - kernel_shape[1]) / stride) + 1)

    if keep_size:
        # Calculate and apply the padding to keep the resultant output shape match the input image shape
        # out_shape = ((shape + (2 * padding) - kernel_shape) / stride) + 1 
        # Rearranged to determine padding_width
        # padding = (stride(out_shape - 1) - shape + kernel_shape) / 2
        # out_shape == image shape as size is maintained  
        padding = int((stride * (image.shape[0] - 1) - image.shape[0] + kernel.shape[0]) / 2.0)
        image = np.pad(image, pad_width=padding, mode=padding_mode)
        output_shape_x = input_image_shape[0]
        output_shape_y = input_image_shape[1]
        
    output = np.zeros((output_shape_x, output_shape_y))
   
    # Convolve the padded image up to the 
    out_x = 0
    for x in range(0, image.shape[0] - kernel_shape[0], stride):
        out_y = 0
        for y in range(0, image.shape[1] - kernel_shape[1], stride):
            data_arr = image[x: x + kernel.shape[0], y: y + kernel.shape[1]]
            sum = np.sum(data_arr * kernel)
            output[out_x, out_y] = sum
            out_y += 1
        out_x +=1
    
    return output


def convolve_2d_cnn_fast(image : np.ndarray, kernel: np.ndarray):
    """
    Applys a convultion kernel to an image in the CNN (separate implementation due to the stacked nature of feature maps and speed requirement).
    Parameters:
        image (np.ndarray): Input image data.
        kernel (np.ndarray): The mask to be convolved across the image.
    Returns:
        image (np.ndarray): The convolved image.
    """
    # Attempting vectorized convolution for CNN
    from numpy.lib.stride_tricks import sliding_window_view

    # Create a sliding window view this is apparently slow on larger images
    # However for 32x32 images it's much (much) faster than loops 10000 convs per second
    sliding_kernel_views = sliding_window_view(image, kernel.shape)

    # Standard convolution, only change is the flattening of the ending dimension for axis based summation
    conv = sliding_kernel_views * kernel
    
    # Flatten the last n dimensions of the result i.e kernel.ndims
    conv = conv.reshape(*conv.shape[:kernel.ndim], -1)
    conv = np.sum(conv, axis=kernel.ndim, keepdims=True)

    # Squeeze any 1 length dimensions out
    conv = np.squeeze(conv)

    return conv

    