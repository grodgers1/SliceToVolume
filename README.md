# SliceToVolume
Histology to microtomography (2d-3d) registration for a rat jaw sample. This is part of a project that was published in "Combining high-resolution hard X-ray tomography and histology for stem cell-mediated distraction osteogenesis" Applied Sciences 12(12) (2022) 6268, DOI: 10.3390/app12126286.

The scripts are written in Matlab, with some 2d-2d registrations done in elastix (https://elastix.lumc.nl/).

# Usage
Follow the script SemiAutomatic_RatJaw.m, which walks through the registration and explains each step.

## Datasets
Please find the microtomography and histology datasets here: https://doi.org/10.5281/zenodo.7189795. They should be placed in example/data/ before running the scripts.

## Notes
2d-2d registrations were done using elastix (https://elastix.lumc.nl/), here the results are given.

# More about slice-to-volume registration
See the review paper by Ferrante and Paragios (10.1016/j.media.2017.04.010) for an overview. The background of this project is work done within the Biomaterials Science Center, University of Basel, which are represented in papers by Chicherova (10.1111/jmi.12692, 10.1007/978-3-319-10404-1_31), Hieber (10.1038/srep32156), and Khimchenko (10.1016/j.neuroimage.2016.06.005).
