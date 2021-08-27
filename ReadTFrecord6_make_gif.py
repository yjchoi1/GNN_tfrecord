import sys
import tensorflow as tf
import os
import numpy as np
import collections
import functools
import json
import os
import pickle
import matplotlib.pyplot as plt
import imageio

def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out

def parse_serialized_simulation_example(example_proto, metadata):
  """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
  if 'context_mean' in metadata:
    feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
  else:
    feature_description = _FEATURE_DESCRIPTION
  context, parsed_features = tf.io.parse_single_sequence_example(
      example_proto,
      context_features=_CONTEXT_FEATURES,
      sequence_features=feature_description)
  for feature_key, item in parsed_features.items():
    convert_fn = functools.partial(
        convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
    parsed_features[feature_key] = tf.py_function(
        convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

  # There is an extra frame at the beginning so we can calculate pos change
  # for all frames used in the paper.
  position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

  # Reshape positions to correct dim:
  parsed_features['position'] = tf.reshape(parsed_features['position'],
                                           position_shape)
  # Set correct shapes of the remaining tensors.
  sequence_length = metadata['sequence_length'] + 1
  if 'context_mean' in metadata:
    context_feat_len = len(metadata['context_mean'])
    parsed_features['step_context'] = tf.reshape(
        parsed_features['step_context'],
        [sequence_length, context_feat_len])
  # Decode particle type explicitly
  context['particle_type'] = tf.py_function(
      functools.partial(convert_fn, encoded_dtype=np.int64),
      inp=[context['particle_type'].values],
      Tout=[tf.int64])
  context['particle_type'] = tf.reshape(context['particle_type'], [-1])
  return context, parsed_features

def read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

metadata = read_metadata('C:/tmp/datasets/Sand/')
print(f'Print metadata: {metadata}')

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

##############################################################################################
# Load serialized law data from 'test.tfrecord'
material_type = 'Sand'
data_type = 'test'
ds_raw = tf.compat.v1.data.TFRecordDataset(f'C:/tmp/datasets/{material_type}/{data_type}.tfrecord')
print(f'Raw data is: {ds_raw}')

# Decode data from 'ds_raw' as 'ds'
ds = ds_raw.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
print(f'Decoded tfrecord is: {ds}')
print(f'Data type is: {type(ds)}')
ds1 = ds.take(2)    # take the first data of the 'ds'
print(f'The first data of ds is: {ds1}')
print(f'Data type is: {type(ds1)}')


## previous reading code ########################

# # Read 'ds' line by line, make a list of them, and save it as a '.txt' file
# decodedDataLines = []
# # sys.stdout = open(f'decodedDataLines_{material_type}.txt', 'w')
# for data_dictionary in ds:
#     decodedDataLines.append(data_dictionary)
# # print(decodedDataLines)
# # sys.stdout.close()
##################################################

## New reading code ########################
decodedDataLines = []
for data_dictionary in ds:
    decodedDataLines.append(data_dictionary)
print(decodedDataLines)
############################################



# Read the number of input data
numberOfKeys = len(decodedDataLines)

# Read the data size of the input data
data1_dict = decodedDataLines[0][1]
data1_positionInfo = data1_dict['position']
data1_array = data1_positionInfo.numpy()
data1_shape = np.shape(data1_array)
numberOfFrames = data1_shape[0]
numberOfParticles = data1_shape[1]
numberOfDimension = data1_shape[2]

# Make a GIF image for showing particle trajectory
for key in range(1):    # substitute i as numberOfKeys later!!

    particle_type = decodedDataLines[key][0]['particle_type']
    particle_type_array = particle_type.numpy()

    position = decodedDataLines[key][1]['position']
    position_array = position.numpy()

    filenames = []

    # Need to update the following line to plot the particle with different markers depending on the particle types
    # Also, it need to consider the size of axis and dimension
    directory = f'C:/Users/baage/Desktop/Choi_MSI/TFRecord-DeepMind/trajectory_{material_type}'
    directory_sub = f'trajectories{key}'
    trajectory_dir = os.path.join(directory, directory_sub)
    os.mkdir(trajectory_dir)

    for frame in range(numberOfFrames):
        x_position = position_array[frame, :, 0]
        y_position = position_array[frame, :, 1]
        plt.figure()
        plt.plot(x_position, y_position, 'bo')
        plt.axis([0, 1, 0, 1])
        filename = f"trajectory{key}_{frame}.png"
        plt.savefig(f'./trajectory/trajectories{key}/{filename}')
        filenames.append(filename)

    with imageio.get_writer(f'mygif{key}.gif', mode='I') as writer:
        for frame in filenames:
            image = imageio.imread(f'./trajectory/trajectories{key}/{frame}')
            writer.append_data(image)



sys.exit()




