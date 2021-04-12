def limit_gpu():
    """This function is used to limit the GPU usage of your tensorflow model.
    
    Change the memory_limit value according to your available memory."""
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
        except RuntimeError as e:
            print(e)
