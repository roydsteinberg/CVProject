from towhee import pipe, ops, DataCollection
import numpy as np
import scipy.io as sio
import torch

"""img_pipe = (
    pipe.input('url')
    .map('url', 'img', ops.image_decode.cv2_rgb())
    .map('img', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='image'))
    .output('img', 'vec')
)"""

model_net40_labels = [
  'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone'
  , 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard'
  , 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio'
  , 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase'
  , 'wardrobe', 'xbox'

]



blip_text_embedding = []
mesh_or_pointclouds = 'point_cloud'

for i, class_name in enumerate(model_net40_labels):
    text_pipe = (
        pipe.input('text')
        .map('text', 'vec', ops.image_text_embedding.blip(model_name='blip_itm_base_coco', modality='text'))
        .output('text', 'vec')
    )
    #text = 'a vertex of a' #'a mesh of a' #+ class_name #"a mesh of a " + class_name
    # text = 'adversarial attack' #'robust features' #'hello new york'  # 'a mesh of a' #+ class_name #"a mesh of a " + class_name
    """if i != 0:
        text = 'misclassify as a' + model_net40_labels[-1] #"a mesh of a " + class_name
    else:
        text = 'a mesh of a' + model_net40_labels[- 1]  # "a mesh of a " + class_name"""
    vowel_check = ''
    if class_name[0] in ['a', 'o', 'i', 'e', 'u', 'x']:
        vowel_check = 'n'
    text = 'a point cloud of a' + vowel_check + ' ' + class_name
    output = DataCollection(text_pipe(text))
    #np.reshape()
    if mesh_or_pointclouds == 'point_cloud':
        blip_text_embedding.append(output[0]['vec'][:-4])
    else:
        blip_text_embedding[i] = np.reshape(output[0]['vec'][:-1], (1, 85,3))
        """blip_text_embedding[str(i)] = np.zeros(shape=(4, 84, 3))
        for j in range(4):
            blip_text_embedding[str(i)][j,:,:] = np.reshape(output[0]['vec'][:-4], (1, 84, 3))"""
    #print(pooler_output.size())

if mesh_or_pointclouds == 'point_cloud':
    blip_final_embed = {}
    blip_final_embed['word'] = np.array(blip_text_embedding)
    sio.savemat('Transductive_ZSL_3D_Point_Cloud/data/ModelNet/BLIPModelNetwordvector.mat', blip_final_embed)
    # np.savez("./blip_text_embedding_no_class_point_clouds.npz", **blip_text_embedding)
    # blip_text_embedding_no_class_6 = np.load('blip_text_embedding_no_class_point_clouds.npz', encoding='latin1', allow_pickle=True)
    # blip_text_embedding_no_class_6 = {k: v for k, v in blip_text_embedding_no_class_segmentation.items()}
else:
    #np.savez("./blip_text_embedding_no_class_mesh_classification_" + text +".npz", **blip_text_embedding)
    np.savez("./blip_text_embedding_adversarial_" + text + ".npz", **blip_text_embedding)


#DataCollection(img_pipe('./teddy.jpg')).show()
