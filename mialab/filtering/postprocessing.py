"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

import numpy as np
# import pydensecrf.densecrf as crf
# import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImagePostProcessing(pymia_fltr.Filter):
    """Post-processing for multi-label brain segmentation.

    Focuses on improving small structures like Hippocampus (3) and Amygdala (4)
    by:
        1) Removing tiny connected components.
        2) Keeping at most two largest blobs for 3 and 4 (left/right).
        3) Applying a small morphological closing on those cleaned blobs.
    """

    def __init__(self,
                 min_size_small: int = 20,
                 min_size_large: int = 10,
                 hip_label: int = 3,
                 amyg_label: int = 4,
                 max_components_small: int = 2):
        """
        Args:
            min_size_small: minimum size (voxels) for hippo/amygdala components.
            min_size_large: minimum size (voxels) for other labels (1, 2, 5).
            hip_label: label id used for hippocampus in the ground truth.
            amyg_label: label id used for amygdala in the ground truth.
            max_components_small: max number of components to keep for
                                  hippocampus/amygdala (typically 2: left/right).
        """
        super().__init__()
        self.min_size_small = min_size_small
        self.min_size_large = min_size_large
        self.hip_label = hip_label
        self.amyg_label = amyg_label
        self.max_components_small = max_components_small

    @staticmethod
    def _keep_components(mask_img: sitk.Image,
                         min_size: int,
                         max_components: int | None = None) -> sitk.Image:
        """Keep connected components above min_size, optionally only the largest N.

        Args:
            mask_img: binary Sitk image (0/1) for a single label.
            min_size: minimum number of voxels per component.
            max_components: if not None, keep only the largest N components.

        Returns:
            A binary Sitk image with only the selected components.
        """
        # Connected components
        cc = sitk.ConnectedComponent(mask_img)

        # Collect stats
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        # Get labels sorted by size (descending)
        labels = list(stats.GetLabels())
        labels_sorted = sorted(
            labels,
            key=lambda lbl: stats.GetNumberOfPixels(lbl),
            reverse=True
        )

        # Filter by size
        labels_filtered = [
            lbl for lbl in labels_sorted
            if stats.GetNumberOfPixels(lbl) >= min_size
        ]

        # Limit to largest N if requested
        if max_components is not None:
            labels_filtered = labels_filtered[:max_components]

        # Rebuild binary mask
        if not labels_filtered:
            # Nothing to keep
            out = sitk.Image(mask_img.GetSize(), sitk.sitkUInt8)
            out.CopyInformation(mask_img)
            return out

        # Start from empty image
        out = sitk.Image(mask_img.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(mask_img)

        for lbl in labels_filtered:
            out = out | (cc == lbl)

        return out

    def execute(self, image: sitk.Image,
                params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Applies post-processing on a multi-label segmentation.

        Steps:
            1) For labels 1,2,5: remove tiny components (min_size_large).
            2) For hippo (3) & amygdala (4):
                - keep only components >= min_size_small
                - keep at most 2 largest blobs (left/right)
                - apply small morphological closing.
        """
        # Convert to numpy for easier label-wise manipulation
        seg_arr = sitk.GetArrayFromImage(image)   # shape (z, y, x)
        out_arr = np.zeros_like(seg_arr, dtype=seg_arr.dtype)

        # Labels you have: 0 BG, 1 WM, 2 GM, 3 Hippo, 4 Amyg, 5 Thalamus
        labels_all = [1, 2, 3, 4, 5]

        for label in labels_all:
            mask = (seg_arr == label).astype(np.uint8)
            if mask.sum() == 0:
                continue

            mask_img = sitk.GetImageFromArray(mask)
            mask_img.CopyInformation(image)

            # Choose parameters depending on label
            if label in [self.hip_label, self.amyg_label]:
                min_size = self.min_size_small
                max_comp = self.max_components_small
            else:
                min_size = self.min_size_large
                max_comp = None  # keep all "large enough" blobs

            # Keep only relevant components
            cleaned_mask = self._keep_components(mask_img,
                                                 min_size=min_size,
                                                 max_components=max_comp)

            # For hippocampus & amygdala, apply a small closing to fill holes
            if label in [self.hip_label, self.amyg_label]:
                cleaned_mask = sitk.BinaryMorphologicalClosing(
                    cleaned_mask,
                    kernelRadius=(1, 1, 1)  # you can try (2,2,2) if needed
                )

            cleaned_arr = sitk.GetArrayFromImage(cleaned_mask) > 0
            out_arr[cleaned_arr] = label

        # Convert back to Sitk
        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(image)
        return out_img

    def __str__(self):
        return 'ImagePostProcessing:\n'.format(self=self)


# class DenseCRFParams(pymia_fltr.FilterParams):
#     """Dense CRF parameters."""
#     def __init__(self, img_t1: sitk.Image, img_t2: sitk.Image, img_proba: sitk.Image):
#         """Initializes a new instance of the DenseCRFParams
#
#         Args:
#             img_t1 (sitk.Image): The T1-weighted image.
#             img_t2 (sitk.Image): The T2-weighted image.
#             img_probability (sitk.Image): The posterior probability image.
#         """
#         self.img_t1 = img_t1
#         self.img_t2 = img_t2
#         self.img_probability = img_probability
#
#
# class DenseCRF(pymia_fltr.Filter):
#     """A dense conditional random field (dCRF).
#
#     Implements the work of Krähenbühl and Koltun, Efficient Inference in Fully Connected CRFs
#     with Gaussian Edge Potentials, 2012. The dCRF code is taken from https://github.com/lucasb-eyer/pydensecrf.
#     """
#
#     def __init__(self):
#         """Initializes a new instance of the DenseCRF class."""
#         super().__init__()
#
#     def execute(self, image: sitk.Image, params: DenseCRFParams = None) -> sitk.Image:
#         """Executes the dCRF regularization.
#
#         Args:
#             image (sitk.Image): The image (unused).
#             params (FilterParams): The parameters.
#
#         Returns:
#             sitk.Image: The filtered image.
#         """
#
#         if params is None:
#             raise ValueError('Parameters are required')
#
#         img_t2 = sitk.GetArrayFromImage(params.img_t1)
#         img_ir = sitk.GetArrayFromImage(params.img_t2)
#         img_probability = sitk.GetArrayFromImage(params.img_probability)
#
#         # some variables
#         x = img_probability.shape[2]
#         y = img_probability.shape[1]
#         z = img_probability.shape[0]
#         no_labels = img_probability.shape[3]
#
#         img_probability = np.rollaxis(img_probability, 3, 0)
#
#         d = crf.DenseCRF(x * y * z, no_labels)  # width, height, nlabels
#         U = crf_util.unary_from_softmax(img_probability)
#         d.setUnaryEnergy(U)
#
#         stack = np.stack([img_t2, img_ir], axis=3)
#
#         # Create the pairwise bilateral term from the above images.
#         # The two `s{dims,chan}` parameters are model hyper-parameters defining
#         # the strength of the location and image content bi-laterals, respectively.
#
#         # higher weight equals stronger
#         pairwise_energy = crf_util.create_pairwise_bilateral(sdims=(1, 1, 1), schan=(1, 1), img=stack, chdim=3)
#
#         # `compat` (Compatibility) is the "strength" of this potential.
#         compat = 10
#         # compat = np.array([1, 1], np.float32)
#         # weight --> lower equals stronger
#         # compat = np.array([[0, 10], [10, 1]], np.float32)
#
#         d.addPairwiseEnergy(pairwise_energy, compat=compat,
#                             kernel=crf.DIAG_KERNEL,
#                             normalization=crf.NORMALIZE_SYMMETRIC)
#
#         # add location only
#         # pairwise_gaussian = crf_util.create_pairwise_gaussian(sdims=(.5,.5,.5), shape=(x, y, z))
#         #
#         # d.addPairwiseEnergy(pairwise_gaussian, compat=.3,
#         #                     kernel=dcrf.DIAG_KERNEL,
#         #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#         # compatibility, kernel and normalization
#         Q_unary = d.inference(10)
#         # Q_unary, tmp1, tmp2 = d.startInference()
#         #
#         # for _ in range(10):
#         #     d.stepInference(Q_unary, tmp1, tmp2)
#         #     print(d.klDivergence(Q_unary) / (z* y*x))
#         # kl2 = d.klDivergence(Q_unary) / (z* y*x)
#
#         # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
#         map_soln_unary = np.argmax(Q_unary, axis=0)
#         map_soln_unary = map_soln_unary.reshape((z, y, x))
#         map_soln_unary = map_soln_unary.astype(np.uint8)  # convert to uint8 from int64
#         # Saving int64 with SimpleITK corrupts the file for Windows, i.e. opening it raises an ITK error:
#         # Unknown component type error: 0
#
#         img_out = sitk.GetImageFromArray(map_soln_unary)
#         img_out.CopyInformation(params.img_t1)
#         return img_out
