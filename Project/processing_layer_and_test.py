'''
Từ khúc này chưa check xong nên bỏ ra 1 file riêng
'''
class PreprocessLayer(tf.keras.layers.Layer):
    '''
    This class represents the PreprocessLayer layer for the Transformer model.
    Methods:
        call(data0, resize=True): Executes the layer's logic.
    Returns:
        tf.Tensor: Output Tensor.
    '''
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    @tf.function(
        input_signature=(tf.TensorSpec(
            shape=[None, N_COLS0], dtype=tf.float32),),
    )
    def call(self, data0, resize=True):
        # Fill NaN Values With 0
        data = tf.where(tf.math.is_nan(data0), 0.0, data0)

        # Hacky
        data = data[None]

        # Empty Hand Frame Filtering
        hands = tf.slice(data, [0, 0, 0], [-1, -1, 84])
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]

        # Pad Zeros
        N_FRAMES = len(data[0])
        if N_FRAMES < N_TARGET_FRAMES:
            data = tf.concat((
                data,
                tf.zeros([1, N_TARGET_FRAMES-N_FRAMES,
                         N_COLS0], dtype=tf.float32)
            ), axis=1)
        # Downsample
        data = tf.image.resize(
            data,
            [1, N_TARGET_FRAMES],
            method=tf.image.ResizeMethod.BILINEAR,
        )

        # Squeeze Batch Dimension
        data = tf.squeeze(data, axis=[0])

        return data


# preprocess_layer = PreprocessLayer()


def load_weight(model, weight_file):
    # Load weights
    model.load_weights(weight_file)


def predict(model, frames):
    phrase_pred = predict_phrase(frames, model).numpy()
    phrase_pred = outputs2phrase(phrase_pred)
    return phrase_pred

