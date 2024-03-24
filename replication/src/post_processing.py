'''
Post-processing in QCANet
We performed (a) reinterpolation and (b) marker-based watershed
transformation on the semantic segmentation image output from NSN
and NDN. Reinterpolation restores the resolution of the image interpolated
for segmentation and identification. Marker-based watershed divides the
semantic segmentation region by watershed with the centre region of the
identified nucleus as a marker. Post-processing enables QCANet to execute
instance segmentation.
'''