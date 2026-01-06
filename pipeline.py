"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings
import pandas as pd
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
from mialab.utilities.pipeline_utilities import mi_feature_selection

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load




def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    # TODO: Modify here
    PyRadiomics_features = {
        'first_order': ["Mean", "Median", "Variance", "Skewness", "Kurtosis", "Energy", "Entropy"], # 7 + 14 = 21 features
        'glcm': ["Contrast", "Correlation", "Idm","JointEntropy", "JointEnergy", "Autocorrelation", "ClusterProminence", "ClusterShade", "DifferenceVariance"], # 7 + 18 = 25 features
        'glrlm': ["ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity", "RunEntropy"], #7 + 8 = 15 features
        'glszm': ["SmallAreaEmphasis", "LargeAreaEmphasis", "ZoneEntropy", "ZoneVariance"], #7 + 8 = 15 features
        'gldm': ["DependenceNonUniformity", "DependenceEntropy", "GrayLevelVariance"], #7 + 6 = 13 features
        'ngtdm': ["Coarseness", "Contrast", "Busyness","Complexity","Strength"], #7 + 10 = 21 features
        'shape': ["MeshVolume","VoxelVolume","SurfaceArea","SurfaceVolumeRatio","Sphericity", "Maximum3DDiameter","Maximum2DDiameterSlice","Maximum2DDiameterColumn","Maximum2DDiameterRow","MajorAxisLength", "MinorAxisLength","LeastAxisLength","Elongation","Flatness"] #35
   }
    
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True,
                          'neighborhood_features': False, #7 + 32 features
                          'PyRadiomics_features': PyRadiomics_features}  #'PyRadiomics_features': PyRadiomics_features
                          
                         

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    #--- DEBUG: print internal structure of first image ---
    print("\n===== DEBUG: images[0].__dict__ =====")
    print(images[0].__dict__)
    print("=====================================\n")

    def build_feature_names(pre_process_params, n_cols):
        names = []

        # 1) Atlas coordinates
        if pre_process_params.get("coordinates_feature", False):
            names += ["ATLAS_COORD_X", "ATLAS_COORD_Y", "ATLAS_COORD_Z"]

        # 2) Intensity (T1, T2)
        if pre_process_params.get("intensity_feature", False):
            names += ["T1_INTENSITY", "T2_INTENSITY"]

        # 3) Gradient (T1, T2)
        if pre_process_params.get("gradient_intensity_feature", False):
            names += ["T1_GRADIENT", "T2_GRADIENT"]

        # 4) PyRadiomics (safe if None)
        pyr = pre_process_params.get("PyRadiomics_features") or {}
        for family, feature_list in pyr.items():
            for modality in ["T1", "T2"]:
                for feat in feature_list:
                    names.append(f"{modality}_RADIOMICS_{family}_{feat}")

        if len(names) != n_cols:
            raise ValueError(
                f"feature_names length ({len(names)}) != X columns ({n_cols}). "
                "Feature stacking order or enabled features don't match naming logic."
            )

        return names

    # attach names into each BrainImage so later code can access them
    n_cols = images[0].feature_matrix[0].shape[1]
    feature_names = build_feature_names(pre_process_params, n_cols)

    for img in images:
        X, y = img.feature_matrix
        img.feature_matrix = (X, y, feature_names)

    # generate feature matrix and label vector

    use_mi_selection = False          # <--- change this to False to use ALL features

    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # ---Print feature table for inspection (before MI selection) ---
    print("data_train shape:", data_train.shape)      # number of samples × number of features
    print("First few rows of data_train:\n", data_train[:5, :])  # first 5 samples
    # Optionally: print feature counts per class
    print("Unique labels in training set:", np.unique(labels_train, return_counts=True))

    # NEW: retrieve feature names from feature_matrix
    feature_names = images[0].feature_matrix[2]

    # NEW: define the 7 core features to analyse
    core_cols = [
        "ATLAS_COORD_X", "ATLAS_COORD_Y", "ATLAS_COORD_Z",
        "T1_INTENSITY", "T2_INTENSITY",
        "T1_GRADIENT", "T2_GRADIENT",
    ]
    # map feature names to column indices
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # safety check (prevents silent bugs)
    missing = [c for c in core_cols if c not in name_to_idx]
    if missing:
        raise ValueError(f"Missing core features: {missing}")

    core_idx = [name_to_idx[c] for c in core_cols]

    # extract ALL rows, ONLY the 7 columns
    X_core = data_train[:, core_idx]   # shape = (n_samples, 7)

    # convert to a labelled table for correlation analysis
    try:
        
        core_table = pd.DataFrame(X_core, columns=core_cols)
    except ImportError:
        core_table = X_core

    print("this is the core table: ")
    print(core_table)
    # correlation matrix (Pearson)
    corr_7x7 = np.corrcoef(X_core, rowvar=False)   # shape: (7, 7)

    try:
        
        corr_table = pd.DataFrame(corr_7x7, index=core_cols, columns=core_cols)

        print("\nCorrelation matrix (Pearson) for core features:")
        print(corr_table.round(3))

        groups = {
            "coordinates": ["ATLAS_COORD_X", "ATLAS_COORD_Y", "ATLAS_COORD_Z"],
            "intensity": ["T1_INTENSITY", "T2_INTENSITY"],
            "gradient": ["T1_GRADIENT", "T2_GRADIENT"],
        }

        group_summary = {}
        for g1, cols1 in groups.items():
            for g2, cols2 in groups.items():
                sub_corr = corr_table.loc[cols1, cols2].abs().values
                group_summary[(g1, g2)] = sub_corr.mean()

        group_corr_table = pd.Series(group_summary).unstack().round(3)

        print("\nGroup-level mean absolute correlation:")
        print(group_corr_table)
    except ImportError:
        print("\nCorrelation matrix (Pearson) for core features:")
        print(np.round(corr_7x7, 3))
    

    #  (optional) MI feature selection
    
    selector = None
    selected_idx = None   # <-- add

    if use_mi_selection:
        # 1) compute MI scores ----
        _, _, _, selector = mi_feature_selection(   # <-- don't need data_train_sel
            X_train=data_train,
            y_train=labels_train,
            X_val=None,
            X_test=None,
            threshold=0.0
        )

        mi_scores = selector.mi_scores

        # get names from the stored feature_matrix tuple
        _, _, feature_names = images[0].feature_matrix
        assert len(mi_scores) == len(feature_names), "MI scores and feature names length mismatch"

        # build MI table (ALL features, before selection)
        df_mi = pd.DataFrame({
            "feature_index": np.arange(len(feature_names)),
            "feature_name": feature_names,
            "MI_score": mi_scores
        }).sort_values("MI_score", ascending=False)

        # save MI table
        mi_csv_path = os.path.join(result_dir, "mi_scores_all_features.csv")
        df_mi.to_csv(mi_csv_path, index=False)
        print(f"[MI] Saved MI scores for all features to: {mi_csv_path}")

        # 2) Guarantee: keep core + top-K radiomics ----
        TOPK_RADIOMICS = 10

        CORE = {
            "ATLAS_COORD_X", "ATLAS_COORD_Y", "ATLAS_COORD_Z",
            "T1_INTENSITY", "T2_INTENSITY",
            "T1_GRADIENT", "T2_GRADIENT"
        }

        df = df_mi.copy()
        df["is_core"] = df["feature_name"].isin(CORE)
        df["is_radiomics"] = df["feature_name"].str.contains("RADIOMICS", na=False)

        core_idx = df.loc[df["is_core"], "feature_index"].tolist()
        missing_core = CORE - set(df.loc[df["is_core"], "feature_name"].tolist())
        if missing_core:
            raise ValueError(f"Missing core features in feature_names: {missing_core}")

        rad_pool = df.loc[df["is_radiomics"]]
        if TOPK_RADIOMICS > len(rad_pool):
            raise ValueError(f"Requested TOPK_RADIOMICS={TOPK_RADIOMICS} but only {len(rad_pool)} radiomics features exist")

        rad_df = rad_pool.nlargest(TOPK_RADIOMICS, "MI_score") if TOPK_RADIOMICS > 0 else rad_pool.head(0)
        rad_idx = rad_df["feature_index"].tolist()

        selected_idx = sorted(set(core_idx + rad_idx))   # <-- final indices used everywhere
        X_train_rf = data_train[:, selected_idx]
        print("selected_data_train shape (core + topK radiomics):", X_train_rf.shape)

        # print selected features
        df_sel = df.loc[df["feature_index"].isin(selected_idx), ["feature_index", "feature_name", "MI_score"]]
        df_sel = df_sel.sort_values("MI_score", ascending=False)

        print(f"\nSelected features (core + top-{TOPK_RADIOMICS} radiomics):")
        for _, r in df_sel.iterrows():
            print(f"[{int(r['feature_index']):2d}] {r['feature_name']:<45s} MI = {r['MI_score']:.6f}")

    else:
        X_train_rf = data_train
        print("Using ALL features:", X_train_rf.shape)
    # -------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    #RF classifier (use features from both image 1 and 2, meaning the same set of features is extracted from both images)
    n_features_rf = X_train_rf.shape[1]
    forest = sk_ensemble.RandomForestClassifier(#max_features=images[0].feature_matrix[0].shape[1],
                                                max_features= "sqrt", #X_train_rf.shape[1],  # number of MI-selected features
                                                n_estimators=100,
                                                max_depth=30,
                                                class_weight = 'balanced',
                                                n_jobs=-1,
                                                random_state = 42)

    start_time = timeit.default_timer()
    forest.fit(X_train_rf, labels_train) #@by me, this allows to use selected fearures for classification
    #forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    images_prediction = []
    images_probabilities = []

    for img in images_test:

        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        # apply same MI selector to test features
        test_features = img.feature_matrix[0]
        # Apply MI selector only if we actually used MI during training
        if use_mi_selection and selector is not None:
            test_features_rf = test_features[:, selected_idx]
        else:
            test_features_rf = test_features



        predictions = forest.predict(test_features_rf)
        probabilities = forest.predict_proba(test_features_rf)
        #predictions = forest.predict(img.feature_matrix[0])
        #probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i],
                        img.images[structure.BrainImageTypes.GroundTruth],
                        img.id_ + '-PP')

        sitk.WriteImage(images_prediction[i],
                        os.path.join(result_dir, img.id_ + '_SEG_atlas.mha'), True)
        sitk.WriteImage(images_post_processed[i],
                        os.path.join(result_dir, img.id_ + '_SEG-PP_atlas.mha'), True)

        # Load original native T1 as reference
        t1_native_path = os.path.join(img.path, 'T1native.nii.gz')
        t1_native = sitk.ReadImage(t1_native_path)

        # Registration transform stored in BrainImage is native -> atlas.
        # We need the inverse: atlas -> native
        tx_native_to_atlas = img.transformation
        tx_atlas_to_native = tx_native_to_atlas.GetInverse()

        # Set up a resampler that maps atlas-space segmentations to native space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t1_native)              # size/spacing/origin/dir of original image
        resampler.SetTransform(tx_atlas_to_native)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) 

        seg_native = resampler.Execute(images_prediction[i])
        seg_pp_native = resampler.Execute(images_post_processed[i])

        # Save native-space segmentations (same dimensions as T1native.nii.gz)
        sitk.WriteImage(seg_native,
                        os.path.join(result_dir, img.id_ + '_SEG_native.nii.gz'), True)
        sitk.WriteImage(seg_pp_native,
                        os.path.join(result_dir, img.id_ + '_SEG-PP_native.nii.gz'), True)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
