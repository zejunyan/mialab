import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk

"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings

# import numpy as np
# import pydensecrf.densecrf as crf
# import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
from scipy.ndimage import generic_filter


class ImagePostProcessing(pymia_fltr.Filter):
    """Represents a post-processing filter.

    Steps:
        1) Remove tiny isolated blobs (connected components smaller than `min_size`)
           for labels 1–5.
        2) For Hippocampus (3) and Amygdala (4), apply a local neighborhood-based
           relabeling: if a voxel is surrounded by many neighbors of that label,
           it is relabeled to that structure (fills small gaps / holes).
    """

    def __init__(self, min_size: int = 3,
                 nhood_threshold_hip: int = 5,
                 nhood_threshold_amyg: int = 5):
        """Initializes a new instance of the ImagePostProcessing class.

        Args:
            min_size (int): Minimum number of voxels for a connected component
                to be kept. Components smaller than this are removed.
            nhood_threshold_hip (int): Minimum number of neighbors with label 3
                (Hippocampus) in a 3x3x3 window to relabel a voxel to 3.
            nhood_threshold_amyg (int): Minimum number of neighbors with label 4
                (Amygdala) in a 3x3x3 window to relabel a voxel to 4.
        """
        super().__init__()
        self.min_size = min_size
        self.nhood_threshold_hip = nhood_threshold_hip
        self.nhood_threshold_amyg = nhood_threshold_amyg

    def _majority_relabel(self, seg_arr: np.ndarray,
                          target_label: int,
                          threshold: int) -> np.ndarray:
        """Relabel voxels to `target_label` if enough neighbors are that label.

        Uses a 3x3x3 neighborhood. If the center voxel is not `target_label`
        and at least `threshold` neighbors are `target_label`, the center voxel
        is relabeled to `target_label`.
        """

        def func(window):
            center = window[len(window) // 2]
            # If it's already the target label, keep it
            if center == target_label:
                return target_label
            # Count neighbors that are the target label
            count = np.sum(window == target_label)
            if count >= threshold:
                return target_label
            else:
                return center

        footprint = np.ones((3, 3, 3), dtype=bool)
        # Apply filter over 3D segmentation
        relabeled = generic_filter(
            seg_arr,
            function=func,
            footprint=footprint,
            mode="nearest"
        )
        return relabeled.astype(seg_arr.dtype)

    def execute(self, image: sitk.Image,
                params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Applies simple post-processing on a multi-label segmentation.

        Strategy:
            1) For each label (1–5):
                * take its binary mask
                * run connected component analysis
                * keep all components with at least `min_size` voxels
            2) For Hippocampus (3) and Amygdala (4):
                * apply neighborhood-based relabeling to fill small gaps
            Background (0) is left implicit.

        Args:
            image (sitk.Image): The predicted label image.
            params (FilterParams): Unused.

        Returns:
            sitk.Image: The post-processed label image.
        """

        # Convert to numpy
        seg_arr = sitk.GetArrayFromImage(image)  # shape (z, y, x)
        out_arr = np.zeros_like(seg_arr, dtype=seg_arr.dtype)

        for label in [1, 2, 3, 4, 5]:

            mask = (seg_arr == label).astype(np.uint8)
            if mask.sum() == 0:
                continue

            mask_img = sitk.GetImageFromArray(mask)
            mask_img.CopyInformation(image)
            
            cc = sitk.ConnectedComponent(mask_img)

            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(cc)

            for comp_label in stats.GetLabels():
                size = stats.GetNumberOfPixels(comp_label)
                if size < self.min_size:
                    continue

                comp_cc = cc == comp_label
                comp_arr = sitk.GetArrayFromImage(comp_cc) > 0

                out_arr[comp_arr] = label

        seg_clean = out_arr.copy()

        seg_clean = self._majority_relabel(
            seg_clean,
            target_label=3,
            threshold=self.nhood_threshold_hip
        )

        seg_clean = self._majority_relabel(
            seg_clean,
            target_label=4,
            threshold=self.nhood_threshold_amyg
        )

        out_img = sitk.GetImageFromArray(seg_clean)
        out_img.CopyInformation(image)
        return out_img

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
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
