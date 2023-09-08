# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.filters import gaussian
import os

path = os.getcwd()
path += "/finalEdgePics"
dir_list = os.listdir(path)
print(dir_list)

def ensure_uint8(img):
    """Convert a float image in range [0, 1] to uint8 image in range [0, 255]."""
    return (img * 255).astype(np.uint8)


def gaussian_blur(image, sigma=1.0):
    """Apply Gaussian blur to the image."""
    return gaussian(image, sigma=sigma)

dict ={}
for i in dir_list:
    dict[i] = []


def align(source, target, window_size=15, sigma=1.0):
    source = gaussian_blur(source, sigma=sigma)
    target = gaussian_blur(target, sigma=sigma)

    min_error = np.inf
    best_offset = (0, 0)
    
    for i in range(-window_size, window_size+1):
        for j in range(-window_size, window_size+1):
            shifted = np.roll(source, shift=(i, j), axis=(0, 1))
            
            # L2 norm error calculation
            error = np.sum((shifted - target) ** 2)
            
            if error < min_error:
                min_error = error
                best_offset = (i, j)

    return best_offset


def multiscale_align(source, target, path, max_scale=5, sigma=1.0):
    original_source = source.copy()
    source_pyramid = [source]
    target_pyramid = [target]

    for i in range(max_scale):
        # Apply Gaussian filter before rescaling
        source = gaussian_blur(source, sigma=sigma)
        target = gaussian_blur(target, sigma=sigma)

        source = sk.transform.rescale(source, 0.5, multichannel=False, anti_aliasing=False)
        target = sk.transform.rescale(target, 0.5, multichannel=False, anti_aliasing=False)
        
        source_pyramid.append(source)
        target_pyramid.append(target)

    offset = (0, 0)
    for idx, (s, t) in enumerate(zip(reversed(source_pyramid), reversed(target_pyramid))):
        s = np.roll(s, shift=(offset[0]*2, offset[1]*2), axis=(0, 1))
       
        # Dynamic window size based on the pyramid level
        current_window_size = 2 * (max_scale - idx)
        current_offset = align(s, t, window_size=current_window_size, sigma=sigma)
        
        offset = (offset[0]*2 + current_offset[0], offset[1]*2 + current_offset[1])
    
    dict[path].append(offset)
    
    print(offset)
    return np.roll(original_source, shift=offset, axis=(0, 1))


for i in dir_list:
    # name of the input file
    imname = 'edges/' + i
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Multiscale alignment
    ag = multiscale_align(g, b, i)
    ar = multiscale_align(r, b, i)

for i in dict.keys():
    str = i + ": "
    print(str)
    print(dict[i])

for i in dir_list:
    # name of the input file
    imname = 'data/' + i
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    ag = np.roll(g, shift=dict[i][0], axis=(0, 1))
    ar = np.roll(r, shift=dict[i][1], axis=(0, 1))
    im_out = np.dstack([ar, ag, b])
    im_out_uint8 = ensure_uint8(im_out)
    fname = 'finalEdgePics/'+ i
    skio.imsave(fname, im_out)
    # display the image
    skio.imshow(im_out)
    skio.show()