# Zero Shot Learning on 3D Point Clouds

The aim of this project was to improve upon the ZSL network shown in Cheraghian et al. (2019) [[Paper]](https://arxiv.org/abs/1912.07161) [[git]](https://github.com/ali-chr/Transductive_ZSL_3D_Point_Cloud)

To do so 3 different methods were attempted:

- Changing the classifier to CurveNet.
- Modifying CurveNet's training data to include a text embedding as well.
- Substituting the text embedding with either [BLIP](https://arxiv.org/abs/2201.12086), [CLIP](https://arxiv.org/abs/2103.00020), or a truncated version of CLIP.

# Getting Started

### Installation
- Clone this repo:
```bash
git clone https://github.com/roydsteinberg/CVProject
cd CVProject
```

### Training CurveNet

A trained version of our version of CurveNet with final layer removed is included in CurveNet/core/convert/model.pth.

If you wish to retrain CurveNet run CurveNet/core/main_cls.py with no arguments. Note this saves the model in a directory above the current one.

If you wish to evaluate CurveNet run CurveNet/core/main_cls.py with the arguments: "--eval True --model_path {MODEL_PATH}" where MODEL_PATH is your trained model's path.

After training CurveNet the final layer must be removed to be used in ZSL. To do so run CurveNet/core/check_model.py after changing the hard-coded model path.

After this layer is removed we can extract the feature vector. To do so run CurveNet/core/data_remake.py. This feature vector is used in 


### Training CurveNet with BLIP-injected data

This is the same as the previous section with small changes:
- The pre-trained, final-layer-cut version is incldued in CurveNet/core/convert/model_BLIP.pth.
- To retrain, add the argument "--exp_name {BLIP_XXX}" where XXX can be any user input.
- After removing the final layer run CurveNet/core/data_remake.py but change the hardcoded paths so that they are in a directory called CurveNet_BLIP adjacent to the CurveNet folder in Transductive_ZSL_3D_Point_Cloud\data\ModelNet.


### Running ZSL

Three files are integral to training/evaluating a ZSL network. These are Transductive_ZSL_3D_Point_Cloud/eval.py, Transductive_ZSL_3D_Point_Cloud/train_inductive.py, Transductive_ZSL_3D_Point_Cloud/train_transductive.py.
As their names state, they are used for evaluating, training an inductive network, and training a transductive network respectively.

The type of classifier/data the ZSL network is given is mainly up to the user and is controlled by the arguments. These arguments are:
- dataset: Use only ModelNet.
- backbone: Choose between CurveNet/CurveNet_BLIP/PointConv(default).
- config_path: Should always be Transductive_ZSL_3D_Point_Cloud/config/ModelNet_config.yaml. Modify this file to change the hyper-parameters.
- model_path: In evalutaion this is the specific path to the trained network to be evaluated. In training, this is the directory in which a trained inductive model already exists (for transductive learning), and/or where to save the model (both learning types).
- settings: Used only in evaluation. Either inductive or transductive.
- wordvec_method: Which text embedding is used. Keep empty for the default Word2Vec.

For example, the following trains a transductive version of CurveNet, with BLIP as its text embedding:
```bash
python Transductive_ZSL_3D_Point_Cloud/train_transductive.py --dataset ModelNet --backbone CurveNet --config_path Transductive_ZSL_3D_Point_Cloud/config/ModelNet_config.yaml --model_path Transductive_ZSL_3D_Point_Cloud/saved_model/CurveNet/model_CurveNet_ours_transductive_BLIP.pth --settings transductive --wordvec_method BLIP
```

And to evaluate this model one will use:
```bash
python Transductive_ZSL_3D_Point_Cloud/eval.py --dataset ModelNet --backbone CurveNet --config_path Transductive_ZSL_3D_Point_Cloud/config/ModelNet_config.yaml --model_path Transductive_ZSL_3D_Point_Cloud/saved_model/CurveNet/ --wordvec_method BLIP
```

Note that models are saved automatically when training and the naming convention stems from the arguments used.


### Using Different Text Encoders

All text embeds are included and saved in the Transductive_ZSL_3D_Point_Cloud/data/ModelNet/ directory and hold the embeddings of all labels inside.

To change between them simply use the aforementioned argument "--wordvec_method" when training.


# Running all possibilites

Included in the project is the file Transductive_ZSL_3D_Point_Cloud/train_eval_all.py.

This file goes over all of the interesting combinations and trains the ZSL network once for each version, outputting the results of each and saving its model in a generated directory called "Logs/".


# Visualizing Data

Included in the project is the file Transductive_ZSL_3D_Point_Cloud/tSNE.py which can be used to visualize the feature vectors of each classifier.

The arguments for dataset must be set to ModelNet, and the backbone to the relevant classifier, and an image of the tSNE is outputted to the main folder.
