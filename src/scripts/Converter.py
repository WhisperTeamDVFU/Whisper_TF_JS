import h5py
import numpy as np
import torch
import tensorflow as tf
import json
from os import path

def convertation(filename: str):
    file = torch.load(filename)
    
    for weight in list(file['model_state_dict'].keys()):
        if (len(file['model_state_dict'][weight].shape) == 2) and weight not in ['decoder.positional_embedding', 'encoder.positional_embedding', 'decoder.token_embedding.weight']:
            file['model_state_dict'][weight] = tf.transpose(file['model_state_dict'][weight])
            
        elif (len(file['model_state_dict'][weight].shape) == 3) and (weight in ['encoder.conv2.weight', 'encoder.conv1.weight']):
            file['model_state_dict'][weight] = tf.transpose(file['model_state_dict'][weight], perm=[2, 1, 0])
    
    with h5py.File(path.splitext(filename)[0] + '.hdf5', 'w') as f:
        for key in list(file['model_state_dict'].keys()):
            f.create_dataset(key, data=file['model_state_dict'][key], dtype='float32')

def save_dims_to_json(filename: str):
    data = torch.load(filename)['dims']
    
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file)