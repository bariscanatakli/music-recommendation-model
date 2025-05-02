import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def create_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Create a ResNet-inspired model for music genre classification
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # First residual block
    shortcut = x
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second block - increasing filters
    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third block
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Classification block
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model


def create_efficient_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Create a lightweight MobileNet-inspired model for music genre classification
    """
    model = Sequential()
    
    # Depthwise separable convolutions for efficiency
    # First block
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Depthwise separable conv block 1
    model.add(tf.keras.layers.DepthwiseConv2D((3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(64, (1, 1), padding='same'))  # Pointwise conv
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Depthwise separable conv block 2
    model.add(tf.keras.layers.DepthwiseConv2D((3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(128, (1, 1), padding='same'))  # Pointwise conv
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Classification block
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model


def create_transformer_hybrid(input_shape=(128, 128, 1), num_classes=8):
    """
    Create a hybrid CNN-Transformer model for music genre classification
    """
    inputs = Input(shape=input_shape)
    
    # CNN feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Prepare for transformer - reshape to sequence
    batch_size = tf.shape(x)[0]
    height = x.shape[1]
    width = x.shape[2]
    channels = x.shape[3]
    
    # Reshape to sequence format
    x = tf.reshape(x, [batch_size, height * width, channels])
    
    # Transformer layers
    transformer_units = 64
    x = tf.keras.layers.Dense(transformer_units, activation='relu')(x)
    
    # Transformer encoder blocks
    for _ in range(2):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=transformer_units // 4)(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(transformer_units * 2, activation='relu'),
            tf.keras.layers.Dense(transformer_units)
        ])
        ffn_output = ffn(x)
        x = tf.keras.layers.Add()([x, ffn_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling over sequence length
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model