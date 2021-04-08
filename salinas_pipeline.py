from salinas_utils import load_data
from salinas_utils.s_pipelines import u_net_pipeline, u_net_sep_pipeline, cnn_1d_pipeline
import tensorflow as tf 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":

    activate_pca = True  # True or False
    pca_explained_variance = 0.9999 # Explained variance required


    t_d, t_l, v_d, v_l, te_d, te_l, classes, weights = load_data.load()
    accuracy, precision, recall, f1 = [], [], [], []
    for i in range(0, 100):
        print("This is the ", i, "-th run")
        a, p, r, f = u_net_sep_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        f1.append(f)

    import numpy as np 
    print("Accuracy : ", sum(accuracy)/len(accuracy), np.std(accuracy))
    print("Precision : ", sum(precision)/len(precision), np.std(precision))
    print("Recall : ", sum(recall)/len(recall), np.std(recall))
    print("F1-score :", sum(f1)/len(f1), np.std(f1))
    #u_net_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)
    #cnn_1d_pipeline(t_d, t_l, v_d, v_l, te_d, te_l, classes, weights)

    print("OK")

