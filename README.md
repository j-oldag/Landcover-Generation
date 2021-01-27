# Landcover-Generation
Landcover mapping created from SAR multiband imagery. Utilizes a Deep UNET architecture (Weights and model not included).
- Deep UNET architecture modified from that by Sumit Kumar (https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation).
- Model trained on data from the Chesapeake Landcover Mapping Project (https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/land-cover-data-project/) with a custom classification schema.

### How it works...
- composites.py begins by creating 3 derived layers (NDVI, EVI, and texture), before stacking all layers to create 7-band imagery from SAR Aerial Imagery inputs.
- prediction.py feeds these bands into a Deep UNET model, created in Keras. This returns 7 prediction layers with prediction confidence recorded for each pixel.
- classify.py assigns a landcover class to each pixel based on the class confidence. Finally, all images are merged together to form a county-level landcover map.
