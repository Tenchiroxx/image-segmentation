from salinas_utils import load_data
from salinas_utils.s_pipelines import u_net_pipeline, u_net_sep_pipeline, cnn_1d_pipeline
import tensorflow as tf 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":

    activate_pca = True  # True or False
    pca_explained_variance = 0.9999 # Explained variance required


    t_d, t_l, v_d, v_l, te_d, te_l, classes, weights = load_data.load()

    #u_net_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #u_net_sep_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #cnn_1d_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)

    print("OK")

