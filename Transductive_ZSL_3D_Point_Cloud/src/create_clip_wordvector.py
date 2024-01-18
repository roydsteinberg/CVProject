from autogluon.multimodal import MultiModalPredictor
import numpy as np
import scipy.io as sio
import torch

model_net40_labels = [
  'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone'
  , 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard'
  , 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio'
  , 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase'
  , 'wardrobe', 'xbox'

]



clip_diminished_text_embedding = []
clip_text_embedding = []
mesh_or_pointclouds = 'point_cloud'

for i, class_name in enumerate(model_net40_labels):
    predictor = MultiModalPredictor(hyperparameters={"model.names": ["clip"]}, problem_type="zero_shot")
    vowel_check = ''
    if class_name[0] in ['a', 'o', 'i', 'e', 'u', 'x']:
        vowel_check = 'n'
    text = ['a point cloud of a' + vowel_check + ' ' + class_name]
    output = predictor.extract_embedding({"text": text})
    
    clip_diminished_text_embedding.append(output['text'][0][:300])
    clip_text_embedding.append(output['text'][0])

clip_final_embed = {}
clip_final_embed['word'] = np.array(clip_text_embedding)
clip_diminished_final_embed = {}
clip_diminished_final_embed['word'] = np.array(clip_diminished_text_embedding)
sio.savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CLIPModelNetwordvector.mat', clip_final_embed)
sio.savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/CLIPDiminishedModelNetwordvector.mat', clip_diminished_final_embed)
