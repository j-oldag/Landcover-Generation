# Main module to execute landcover generation functions.

import composites
import prediction
import classify
import time
from multiprocessing import Pool

# Set variables.
county_list = ['Walker','Chattooga','Floyd','Catoosa','Whitfield','Gordon','Bartow','Murray','Fannin','Gilmer','Pickens','Cherokee','Union']

if __name__ == '__main__':

    # Record processing time.
    t0 = time.time()

    # Create pool to divide county landcover generations.
    pool = Pool()

    # Composite separate SAR Imagery rasters into single image.
    pool.map(composites.main,county_list)

    # Feed composite imagery to UNET, output classification confidence layers.
    for c in county_list:
        prediction.main(c)

    # Classify landcover types by confidence thresholds.
    # Merge tiles into county-wide raster.
    pool.map(classify.main,county_list)

    t1 = time.time()
    print('Time elapsed: ',t1-t0)

# Non-multi-core processing.
# for c in county_list:
#     t0 = time.time()
#     composites.main(c)
#
#     print('Time elapsed: ',time.time()-t0)
