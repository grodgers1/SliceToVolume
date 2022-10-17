# SliceToVolume
Histology to microtomography (2d-3d) registration for a rat jaw sample. This is part of a project that was published in "Combining high-resolution hard X-ray tomography and histology for stem cell-mediated distraction osteogenesis" Applied Sciences 12(12) (2022) 6268, DOI: [10.3390/app12126286](https://doi.org/10.3390/app12126286).

The scripts are written in Matlab, with some 2d-2d registrations done in elastix (https://elastix.lumc.nl/).

# Usage
Follow the script SemiAutomatic_RatJaw.m, which walks through the registration and explains each step.

## Datasets
Please find the microtomography and histology datasets here: https://doi.org/10.5281/zenodo.7189795. They should be placed in example/data/ before running the scripts.

## Notes
2d-2d registrations were done using elastix (https://elastix.lumc.nl/), here the results are given.

# Background
## Approach
The approach is described in [C. Tanner et al. "Registration of microtomography images: challenges and approaches "](https://doi.org/10.1117/12.2633922) and in the [main paper](https://doi.org/10.3390/app12126286). Briefly, the microtomography data was coarsely aligned and an oral/maxillofacial surgeion manually labelled 50 corresponding anatomical landmarks between the histology and microtomography. The histology slide and approximate location of the cutting plane along with the matched features are shown in the figure below from [10.1117/12.2633922](https://doi.org/10.1117/12.2633922).

![histology and microtomography with matching landmarks](https://github.com/grodgers1/SliceToVolume/blob/main/example/figures/fig_2d3d_p1.png)

These were used to extract a 2d slice and perform an affine landmark-based alignment. Then, an intensity-based non-rigid registration was applied. The accuracy was assessed by visual inspection and distance between landmarks, see figure below from [10.1117/12.2633922](https://doi.org/10.1117/12.2633922).

![landmark distance](https://github.com/grodgers1/SliceToVolume/blob/main/example/figures/fig_2d3d_p2.png)

## More information
See the review paper by [Ferrante and Paragios](https://doi.org/10.1016/j.media.2017.04.010) for an overview. The background of this project is work done within the Biomaterials Science Center, University of Basel, which are represented in [papers](https://doi.org/10.1111/jmi.12692) by [Chicherova](https://doi.org/10.1007/978-3-319-10404-1_31), [Hieber](https://doi.org/10.1038/srep32156), and [Khimchenko](https://doi.org/10.1016/j.neuroimage.2016.06.005).
