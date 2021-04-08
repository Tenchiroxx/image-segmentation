"""
This is the main file to run a pipeline on the Salinas data.
You have to uncomment the pipeline you want to use, then launch the file. 
"""

from landsat_utils import load_data
from landsat_utils.pipelines import u_net_pipeline, u_net_sep_pipeline, cnn_1d_pipeline
import tensorflow as tf 

# Limiting GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":

    # Loading the data
    t_d, t_l, v_d, v_l, te_d, te_l, classes, weights = load_data.load()

    # Select the CNN architecture you want to use
    
    #u_net_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    u_net_sep_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #cnn_1d_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights) 