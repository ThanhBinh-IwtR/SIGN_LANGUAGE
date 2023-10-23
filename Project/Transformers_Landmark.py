'''
- This module defines the LandmarkEmbedding and Embedding classes used in the Transformer model for sign language recognition. 
- It also defines several constants and hyperparameters used in the model.

Classes:
- LandmarkEmbedding: Embeds a landmark using fully connected layers.
- Embedding: Creates embedding for each frame.
'''

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

SEED = 42
# Number of Frames to resize recording to
N_TARGET_FRAMES = 128
DIR_NAME = 'statics'
CHAR_TO_PREDICTION_INDEX = os.path.join(
    DIR_NAME, 'character_to_prediction_index.json')

with open(CHAR_TO_PREDICTION_INDEX, encoding='utf-8') as json_file:
    CHAR2ORD = json.load(json_file)
SOS_TOKEN = len(CHAR2ORD) + 1  # Start Of Sentence
EOS_TOKEN = len(CHAR2ORD) + 2  # End Of Sentence
ORD2CHAR = {j: i for i, j in CHAR2ORD.items()}
# Number of Unique Characters To Predict + Pad Token + SOS Token + EOS Token
N_UNIQUE_CHARACTERSPAD_TOKEN = len(CHAR2ORD)
N_UNIQUE_CHARACTERS0 = len(CHAR2ORD)
N_UNIQUE_CHARACTERS = len(CHAR2ORD) + 1 + 1 + 1
PAD_TOKEN = len(CHAR2ORD)  # Padding
COLUMNS0 = np.load(DIR_NAME+'/COLUMNS0.npy')
N_COLS0 = N_COLS = len(COLUMNS0)
N_TARGET_FRAMES = 128

# Epsilon value for layer normalisation
LAYER_NORM_EPS = 1e-6

# final embedding and transformer embedding size
UNITS_ENCODER = 256
UNITS_DECODER = 256

# Transformer
NUM_BLOCKS_ENCODER = 6
NUM_BLOCKS_DECODER = 3
NUM_HEADS = 6
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
MHA_DROPOUT_RATIO = 0.20
CLASSIFIER_DROPOUT_RATIO = 0.10
# Length of Phrase + EOS Token
MAX_PHRASE_LENGTH = 31 + 1
# Embeds a landmark using fully connected layers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu

MEANS = np.load(DIR_NAME+'/MEANS.npy').reshape(-1)
STDS = np.load(DIR_NAME+'/STDS.npy').reshape(-1)


class LandmarkEmbedding(tf.keras.Model):
    '''
    This class represents the LandmarkEmbedding layer for the Transformer model.
    Args:
        units (int): Number of Units in Model.
        name (str): Name of the layer.
    Attributes:
        supports_masking (bool): Whether the layer supports masking.
        empty_embedding (tf.Variable): Embedding for missing landmark in frame, initizlied with zeros.
        dense (tf.keras.Sequential): Embedding.
    Methods:
        build(input_shape): Builds the layer.
        call(x): Executes the layer's logic.
    '''

    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False,
                                  kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False,
                                  kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )

# Creates embedding for each frame
class Embedding(tf.keras.Model):
    """
    This class represents the Embedding layer for the Transformer model.

    Attributes:
    - supports_masking (bool): Whether the layer supports masking.
    - positional_embedding (tf.Variable): Positional embedding for each frame index.
    - dominant_hand_embedding (LandmarkEmbedding): Embedding layer for Landmarks.

    Methods:
    - build(input_shape): Builds the layer.
    - call(x, training=False): Executes the layer's logic.
    Returns:
        tf.Tensor: Output Tensor.
    """

    def __init__(self):
        super(Embedding, self).__init__()
        self.supports_masking = True

    def build(self, input_shape):
        """
        Build by Initializes positional and dominant hand embeddings.
        Args:
            input_shape: A tuple representing the shape of the input tensor.

        Returns:
            None
        """

        # Positional embedding for each frame index
        self.positional_embedding = tf.Variable(
            name='embedding_positional_encoder',
            initial_value=tf.zeros(
                [N_TARGET_FRAMES, UNITS_ENCODER], dtype=tf.float32),
            trainable=True
        )
        # Embedding layer for Landmarks
        self.dominant_hand_embedding = LandmarkEmbedding(
            UNITS_ENCODER, 'dominant_hand')

    def call(self, x, training=False):
        '''
        Args:
            x (tf.Tensor): Input Tensor.
            training (bool, optional): Whether the model is training. Defaults to False.   
        Returns:
            tf.Tensor: Output Tensor.
        '''
        # Normalize data, return 0 if data = 0
        x = tf.where(tf.math.equal(x, 0.0), 0.0, (x - MEANS) / STDS)
        # Dominant Hand
        x = self.dominant_hand_embedding(x)
        # Add Positional Encoding
        x = x + self.positional_embedding
        return x

# replaced softmax with softmax layer to support masked softmax
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention Layer.

    Args:
        d_model (int): Number of Units in Model.
        n_heads (int): Number of Attention Heads.
        dropout (float): Dropout Rate.
        d_out (int, optional): Number of Units in Output Layer. Defaults to None.

    Returns:
        tf.Tensor: Output Tensor.
    """

    def __init__(self, d_model, n_heads, dropout, d_out=None):
        super(MultiHeadAttention, self).__init__()
        # Number of Units in Model
        self.d_model = d_model
        # Number of Attention Heads
        self.n_heads = n_heads
        # Number of Units in Intermediate Layers
        self.depth = d_model // n_heads
        # Scaling Factor Of Values
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        # Learnable Projections to Depth
        self.wq = self.fused_mha(self.depth)
        self.wk = self.fused_mha(self.depth)
        self.wv = self.fused_mha(self.depth)
        # Output Projection
        self.wo = tf.keras.layers.Dense(
            d_model if d_out is None else d_out, use_bias=False)
        # Softmax Activation Which Supports Masking
        self.softmax = tf.keras.layers.Softmax()
        # Reshaping Of Multiple Attention heads to Single Value
        self.reshape = tf.keras.Sequential([
            tf.keras.layers.Permute([2, 1, 3]),
            tf.keras.layers.Reshape([N_TARGET_FRAMES, self.depth]),
        ])
        # Output Dropout
        self.do = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    # Single dense layer for all attention heads
    def fused_mha(self, dim):
        """
        Single dense layer for all attention heads.

        Args:
            dim (int): Number of Units in Intermediate Layers.

        Returns:
            tf.keras.Sequential: Sequential Model.

        """
        return tf.keras.Sequential([
            # Single dense layer
            tf.keras.layers.Dense(dim, use_bias=False),
            # Reshape to [number of frames, number of attention head, depth]
            tf.keras.layers.Reshape(
                [N_TARGET_FRAMES, self.n_heads, dim // self.n_heads]),
            # Permutate to [number of attention heads, number of frames, depth]
            tf.keras.layers.Permute([2, 1, 3]),
        ])

    def call(self, q, k, v, attention_mask=None, training=False):
        '''
        Args:
            q (tf.Tensor): Query Tensor.
            k (tf.Tensor): Key Tensor.
            v (tf.Tensor): Value Tensor.
            attention_mask (tf.Tensor, optional): Attention Mask. Defaults to None.
            training (bool, optional): Whether the model is training. Defaults to False.
        Returns: 
            tf.Tensor: Output Tensor.
        '''
        # Projections to attention heads
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)
        # Matrix multiply QxK to acquire attention scores
        x = tf.matmul(Q, K, transpose_b=True) * self.scale
        # Softmax attention scores and Multiply with Values
        x = self.softmax(x, mask=attention_mask) @ V
        # Reshape to flatten attention heads
        x = self.reshape(x)
        # Output projection
        x = self.wo(x)
        # Dropout
        x = self.do(x, training=training)
        return x


class Encoder(tf.keras.Model):
    '''
    This class represents the Encoder layer for the Transformer model.

    Args:
        num_blocks (int): Number of Transformer Blocks.
    Methods:
        build(input_shape): Builds the layer.
        get_attention_mask(x_inp): Returns attention mask.
        call(x, x_inp, training=False): Executes the layer's logic.
    Returns:
        tf.Tensor: Output Tensor.
        '''

    def __init__(self, num_blocks):
        super(Encoder, self).__init__(name='encoder')
        self.num_blocks = num_blocks
        self.supports_masking = True

    def build(self, input_shape):
        '''
        Builds the layer.
        Args:
            input_shape: A tuple representing the shape of the input tensor.
        Returns:
            None'''
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(
                UNITS_ENCODER, NUM_HEADS, MHA_DROPOUT_RATIO))
            # Second Layer Normalisation
            self.ln_2s.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(UNITS_ENCODER * MLP_RATIO, activation=GELU,
                                      kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
                tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(UNITS_ENCODER, kernel_initializer=INIT_HE_UNIFORM,
                                      use_bias=False),
            ]))
            # Optional Projection to Decoder Dimension
            if UNITS_ENCODER != UNITS_DECODER:
                self.dense_out = tf.keras.layers.Dense(UNITS_DECODER,
                                                       kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False)
                self.apply_dense_out = True
            else:
                self.apply_dense_out = False

    def get_attention_mask(self, x_inp):
        '''
        Returns attention mask.
        Args:
            x_inp: Input Tensor.
        Returns:
            tf.Tensor: Attention Mask.
        '''
        # Attention Mask
        attention_mask = tf.math.count_nonzero(
            x_inp, axis=[2], keepdims=True, dtype=tf.int32)
        attention_mask = tf.math.count_nonzero(
            attention_mask, axis=[2], keepdims=False)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        return attention_mask

    def call(self, x, x_inp, training=False):
        '''
        Executes the layer's logic.
        Args:
            x (tf.Tensor): Input Tensor.
            x_inp (tf.Tensor): Input Tensor.
            training (bool, optional): Whether the model is training. Defaults to False.
        Returns:
            tf.Tensor: Output Tensor.
        '''
        # Attention mask to ignore missing frames
        attention_mask = self.get_attention_mask(x_inp)
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(x + mha(x, x, x, attention_mask=attention_mask))
            x = ln_2(x + mlp(x))

        # Optional Projection to Decoder Dimension
        if self.apply_dense_out:
            x = self.dense_out(x)

        return x

# Decoder based on multiple transformer blocks


class Decoder(tf.keras.Model):
    '''
    This class represents the Decoder layer for the Transformer model.
    Args:
        num_blocks (int): Number of Transformer Blocks.
    Methods:
        build(input_shape): Builds the layer.
        get_attention_mask(x_inp): Returns attention mask.
        call(x, encoder_outputs, x_inp, training=False): Executes the layer's logic.
    Returns:
        tf.Tensor: Output Tensor.
    '''

    def __init__(self, num_blocks):
        super(Decoder, self).__init__(name='decoder')
        self.num_blocks = num_blocks
        self.supports_masking = True

    def build(self, input_shape):
        '''
        Builds the Decoder layer.
        Args:
            input_shape: A tuple representing the shape of the input tensor.
        Returns:
            None
        '''
        # Causal Mask Batch Size 1
        self.causal_mask = self.get_causal_attention_mask()
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros(
                [N_TARGET_FRAMES, UNITS_DECODER], dtype=tf.float32),
            trainable=True, name='embedding_positional_encoder'
        )
        # Character Embedding
        self.char_emb = tf.keras.layers.Embedding(N_UNIQUE_CHARACTERS, UNITS_DECODER,
                                                  embeddings_initializer=INIT_ZEROS)
        # Positional Encoder MHA
        self.pos_emb_mha = MultiHeadAttention(
            UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO)
        self.pos_emb_ln = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPS)
        # First Layer Normalisation
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(
                UNITS_DECODER, NUM_HEADS, MHA_DROPOUT_RATIO))
            # Second Layer Normalisation
            self.ln_2s.append(
                tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(UNITS_DECODER * MLP_RATIO, activation=GELU,
                                      kernel_initializer=INIT_GLOROT_UNIFORM, use_bias=False),
                tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(UNITS_DECODER, kernel_initializer=INIT_HE_UNIFORM,
                                      use_bias=False),
            ]))

    def get_causal_attention_mask(self):
        '''
        Returns attention mask.
        Args:
            None
        Returns:
            tf.Tensor: Attention Mask.
        '''
        i = tf.range(N_TARGET_FRAMES)[:, tf.newaxis]
        j = tf.range(N_TARGET_FRAMES)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, N_TARGET_FRAMES, N_TARGET_FRAMES))
        mult = tf.concat(
            [tf.expand_dims(1, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        mask = tf.tile(mask, mult)
        mask = tf.cast(mask, tf.float32)
        return mask

    def get_attention_mask(self, x_inp):
        '''
        Returns attention mask.
        Args:
            x_inp: Input Tensor.
        Returns:
            tf.Tensor: Attention Mask.
        '''
        # Attention Mask
        attention_mask = tf.math.count_nonzero(
            x_inp, axis=[2], keepdims=True, dtype=tf.int32)
        attention_mask = tf.math.count_nonzero(
            attention_mask, axis=[2], keepdims=False)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        return attention_mask

    def call(self, encoder_outputs, phrase, x_inp, training=False):
        '''
        Executes the layer's logic.
        Args:
            encoder_outputs (tf.Tensor): Encoder Output Tensor.
            phrase (tf.Tensor): Phrase Tensor.
            x_inp (tf.Tensor): Input Tensor.
            training (bool, optional): Whether the model is training. Defaults to False.
        Returns:
            tf.Tensor: Output Tensor.
        '''
        # Batch Size
        B = tf.shape(encoder_outputs)[0]
        # Cast to INT32
        phrase = tf.cast(phrase, tf.int32)
        # Prepend SOS Token
        phrase = tf.pad(phrase, [[0, 0], [1, 0]], constant_values=SOS_TOKEN,
                        name='prepend_sos_token')
        # Pad With PAD Token
        phrase = tf.pad(phrase, [[0, 0], [0, N_TARGET_FRAMES-MAX_PHRASE_LENGTH-1]],
                        constant_values=PAD_TOKEN, name='append_pad_token')
        # Positional Embedding
        x = self.positional_embedding + self.char_emb(phrase)
        # Causal Attention
        x = self.pos_emb_ln(x + self.pos_emb_mha(x, x, x,
                            attention_mask=self.causal_mask))
        # Attention mask to ignore missing frames
        attention_mask = self.get_attention_mask(x_inp)
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x = ln_1(x + mha(x, encoder_outputs, encoder_outputs,
                     attention_mask=attention_mask))
            x = ln_2(x + mlp(x))
        # Slice 31 Characters
        x = tf.slice(x, [0, 0, 0], [-1, MAX_PHRASE_LENGTH, -1])
        return x

# Causal Attention to make decoder not attent to future characters which it needs to predict
def get_causal_attention_mask(B):
    '''
    Returns attention mask.
    Args:
        B (int): Batch Size.
    Returns:
        tf.Tensor: Attention Mask.
    '''
    i = tf.range(N_TARGET_FRAMES)[:, tf.newaxis]
    j = tf.range(N_TARGET_FRAMES)
    mask = tf.cast(i >= j, dtype=tf.int32)
    mask = tf.reshape(mask, (1, N_TARGET_FRAMES, N_TARGET_FRAMES))
    mult = tf.concat(
        [tf.expand_dims(B, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    mask = tf.tile(mask, mult)
    mask = tf.cast(mask, tf.float32)
    return mask

# TopK accuracy for multi dimensional output
class TopKAccuracy(tf.keras.metrics.Metric):
    '''
    This class represents the TopKAccuracy metric for the Transformer model.
    Args:
        k (int): Number of Top Predictions.
    Methods:
        update_state(y_true, y_pred, sample_weight=None): Accumulates metric statistics.
        result(): Computes and returns the metric value tensor.
        reset_state(): Resets all of the metric state variables.
    Returns:
        tf.Tensor: Output Tensor.
    '''

    def __init__(self, k, **kwargs):
        super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
        self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Accumulates metric statistics.
        Args:
            y_true (tf.Tensor): True Labels.
            y_pred (tf.Tensor): Predicted Labels.
            sample_weight (tf.Tensor, optional): Sample Weights. Defaults to None.
        Returns:
            None'''
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, N_UNIQUE_CHARACTERS])
        character_idxs = tf.where(y_true < N_UNIQUE_CHARACTERS0)
        y_true = tf.gather(y_true, character_idxs, axis=0)
        y_pred = tf.gather(y_pred, character_idxs, axis=0)
        self.top_k_acc.update_state(y_true, y_pred)

    def result(self):
        '''
        Computes and returns the metric value tensor.
        Args:
            None
        Returns:
            tf.Tensor: Output Tensor.
        '''
        return self.top_k_acc.result()

    def reset_state(self):
        '''
        Resets all of the metric state variables.
        Args:
            None
        Returns:
            None'''
        self.top_k_acc.reset_state()


# Create Initial Loss Weights All Set To 1
loss_weights = np.ones(N_UNIQUE_CHARACTERS, dtype=np.float32)
# Set Loss Weight Of Pad Token To 0
loss_weights[PAD_TOKEN] = 0

def scce_with_ls(y_true, y_pred):
    '''
    Sparse Categorical Crossentropy With Label Smoothing.
    Args:
        y_true (tf.Tensor): True Labels.
        y_pred (tf.Tensor): Predicted Labels.
    Returns:
        tf.Tensor: Output Tensor.
    '''
    # Filter Pad Tokens
    idxs = tf.where(y_true != PAD_TOKEN)
    y_true = tf.gather_nd(y_true, idxs)
    y_pred = tf.gather_nd(y_pred, idxs)
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, N_UNIQUE_CHARACTERS, axis=1)
    # Categorical Crossentropy with native label smoothing support
    loss = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=0.25, from_logits=True)
    loss = tf.math.reduce_mean(loss)
    return loss

def get_model():
    '''
    Returns the Transformer model.
    Args:
        None
    Returns:
        tf.keras.Model: Transformer Model.
    '''
    # Inputs x =  (batch,128,164) encoder
    frames_inp = tf.keras.layers.Input(
        [N_TARGET_FRAMES, N_COLS], dtype=tf.float32, name='frames')
    phrase_inp = tf.keras.layers.Input(
        [MAX_PHRASE_LENGTH], dtype=tf.int32, name='phrase')
    # Frames
    x = frames_inp

    # Masking
    x = tf.keras.layers.Masking(
        mask_value=0.0, input_shape=(N_TARGET_FRAMES, N_COLS))(x)

    # Embedding
    x = Embedding()(x)

    # Encoder Transformer Blocks
    x = Encoder(NUM_BLOCKS_ENCODER)(x, frames_inp)

    # Decoder
    x = Decoder(NUM_BLOCKS_DECODER)(x, phrase_inp, frames_inp)

    # Classifier
    x = tf.keras.Sequential([
        # Dropout
        tf.keras.layers.Dropout(CLASSIFIER_DROPOUT_RATIO),
        # Output Neurons
        tf.keras.layers.Dense(N_UNIQUE_CHARACTERS, activation=tf.keras.activations.linear,
                              kernel_initializer=INIT_HE_UNIFORM, use_bias=False),
    ], name='classifier')(x)

    outputs = x

    # Create Tensorflow Model
    model = tf.keras.models.Model(
        inputs=[frames_inp, phrase_inp], outputs=outputs)

    # Categorical Crossentropy Loss With Label Smoothing
    loss = scce_with_ls

    # Adam Optimizer
    optimizer = tfa.optimizers.RectifiedAdam(sma_threshold=4)
    optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=5)

    # TopK Metrics
    metrics = [
        TopKAccuracy(1),
        TopKAccuracy(5),
    ]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        loss_weights=loss_weights,
    )

    return model

# Output Predictions to string
def outputs2phrase(outputs):
    '''
    Converts model outputs to string.
    Args:
        outputs (np.ndarray): Model Outputs.
    Returns:
        str: String.
    '''
    if outputs.ndim == 2:
        outputs = np.argmax(outputs, axis=1)

    return ''.join([ORD2CHAR.get(s, '') for s in outputs])


@tf.function(jit_compile=True)
def predict_phrase(frames, model):
    '''
    Predicts phrase from frames.
    Args:
        frames (tf.Tensor): Frames Tensor.
        model (tf.keras.Model): Model.
    Returns:
        tf.Tensor: Output Tensor.'''
    # Add Batch Dimension
    frames = tf.expand_dims(frames, axis=0)
    # Start Phrase
    phrase = tf.fill([1, MAX_PHRASE_LENGTH], PAD_TOKEN)

    for idx in tf.range(MAX_PHRASE_LENGTH):
        # Cast phrase to int8
        phrase = tf.cast(phrase, tf.int8)
        # Predict Next Token
        outputs = model({
            'frames': frames,
            'phrase': phrase,
        })

        # Add predicted token to input phrase
        phrase = tf.cast(phrase, tf.int32)
        phrase = tf.where(
            tf.range(MAX_PHRASE_LENGTH) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )

    # Squeeze outputs
    outputs = tf.squeeze(phrase, axis=0)
    outputs = tf.one_hot(outputs, N_UNIQUE_CHARACTERS)

    # Return a dictionary with the output tensor
    return outputs
