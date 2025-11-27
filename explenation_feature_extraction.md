# Explanation Feature Extraction in pipeline

## Where happens what?

### In `pipeline.py:main`

- `pre_process_params`: define what features to activate

- `images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)`:
Here the preprocess for the training happens. Part of the preprocess is the Feature extraction

- `images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)':
Same but for Testing images

### In `mialab/utilities/pipeline_utilities`
`mialab/utilities/pipeline_utilities:pre_process_batch` >>

`mialab/utilities/pipeline_utilities:pre_process` >>

```[python]
    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()
```

`mialab/utilities/pipeline_utilities:FeatreExtractor::execute` >>

Here We can add more features.

## What Features are already implemented

### 1. Atlas Coordinates Feature                                             
                                                                              
- Purpose: Provides spatial location information of each voxel within a     
  standardized coordinate system (atlas space)                                
- Activation: Controlled by  coordinates_feature  parameter                 
- Implementation: Uses  AtlasCoordinates()  filter to transform image       
  coordinates to a standardized brain atlas space                             
- Use case: Helps the model learn spatial relationships and anatomical      
  locations                                                                   
                                                                              
### 2. Intensity Features                                                    
                                                                              
- Purpose: Uses raw image intensity values as features                      
- Activation: Controlled by  intensity_feature  parameter                   
- Implementation: Directly uses T1-weighted and T2-weighted MRI intensities 
- Use case: Basic tissue contrast information from different MRI sequences  
                                                                              
### 3. Gradient Intensity Features                                           
                                                                              
- Purpose: Captures edge information and texture variations                 
- Activation: Controlled by  gradient_intensity_feature  parameter          
- Implementation: Computes gradient magnitude using SimpleITK's             
  GradientMagnitude()  filter                                                 
- Use case: Helps identify boundaries between different tissue types and    
  structural edges  

## What we will implement in a first step

In `mialab/filtering/feature_extraction.py`: 

Here we can use the `NeighborhoodFeatureExtractor` to extract all the first
order features.

Which already implements most of the first order features in pyradiomics

*see: AtlasCoordinates Feature* which is already implemented in a similar way

