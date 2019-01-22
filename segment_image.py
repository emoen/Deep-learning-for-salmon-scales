#Segment salom scales

from skimage import io
from scipy import ndimage
import sys
import os
import glob
from skimage import filters, segmentation
from skimage.measure import label
from skimage.measure import regionprops

from collections import Counter

def do_segment():
    files = glob.glob(os.path.join("./","*.tif"))

    list_regions = []
    counter = 0
for image_file in files:
    counter = counter +1
    if counter > 20: 
        break
        
    im = ndimage.imread(image_file, flatten=True)
    
    # find a dividing line between 0 and 255
    # pixels below this value will be black
    # pixels above this value will be white
    val = filters.threshold_otsu(im)
    
    # the mask object converts each pixel in the image to True or False
    # to indicate whether the given pixel is black/white
    mask = im < val
    
    # apply the mask to the image object
    clean_border = segmentation.clear_border(mask)
    
    # labeled contains one integer for each pixel in the image,
    # where that image indicates the segment to which the pixel belongs
    labeled = label(clean_border)
    
    # create array in which to store cropped articles
    cropped_images = []
    
    # define amount of padding to add to cropped image
    pad = 20
    
    # for each segment number, find the area of the given segment.
    # If that area is sufficiently large, crop out the identified segment
    for region_index, region in enumerate(regionprops(labeled)):
        if region.area < 1000:
            continue
               
        # draw a rectangle around the segmented articles
        # bbox describes: min_row, min_col, max_row, max_col            
        minr, minc, maxr, maxc = region.bbox
        
        # use those bounding box coordinates to crop the image
        cropped_images.append(im[minr-pad:maxr+pad, minc-pad:maxc+pad])
    
    print(str(len(cropped_images))+ ":" + str(image_file) )    
    list_regions.append(len(cropped_images))    

    c = Counter( list_regions )
    print(c.items())

if __name__ == '__main__':
    do_segment()
