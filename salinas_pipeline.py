"""
This is the main file to run a pipeline on the Salinas data.
You have to uncomment the pipeline you want to use, then launch the file. 
"""

from salinas_utils import load_data
from salinas_utils.s_pipelines import u_net_pipeline, u_net_sep_pipeline, cnn_1d_pipeline
import tensorflow as tf 

# Limiting GPU use
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)
        

if __name__ == "__main__":

    # Should a PCA be applied on the input data ?
    activate_pca = True  # True or False
    pca_explained_variance = 0.9999 # Explained variance required

    # Loading the data
    t_d, t_l, v_d, v_l, te_d, te_l, classes, weights = load_data.load()

    # Selecting the architecture to use

    u_net_sep_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #u_net_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #cnn_1d_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)

