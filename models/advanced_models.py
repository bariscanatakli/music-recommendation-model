import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def create_improved_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Create an improved hybrid model for music genre classification
    that combines elements of ResNet and EfficientNet
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution block with increased filters
    x = Conv2D(48, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(1e-5))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # First residual block
    shortcut = x
    x = Conv2D(48, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (3, 3), padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second block - squeeze and excitation
    shortcut = Conv2D(96, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(96, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Conv2D(96, (3, 3), padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    
    # Squeeze and Excitation block
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = Dense(96 // 4, activation='relu')(se)
    se = Dense(96, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, 96))(se)
    x = tf.keras.layers.Multiply()([x, se])
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third block with attention
    shortcut = Conv2D(192, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(192, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), padding='same',
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global features
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layers with balanced dropout
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Slightly reduced dropout
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer and slightly lower learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.00008),
        metrics=['accuracy']
    )
    
    return model

def create_ensemble_model(input_shape=(128, 128, 1), num_classes=8):
    """
    Create an ensemble model that combines multiple pathways for improved accuracy
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Path 1: Standard CNN pathway
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(64, (5, 5), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    
    # Path 2: Frequency-focused pathway (vertical filters)
    x2 = Conv2D(32, (7, 3), activation='relu', padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(64, (5, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    
    # Path 3: Time-focused pathway (horizontal filters)
    x3 = Conv2D(32, (3, 7), activation='relu', padding='same')(inputs)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Conv2D(64, (3, 5), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
    
    # Combine pathways
    combined = tf.keras.layers.Concatenate()([x1, x2, x3])
    
    # Fully connected layers
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

# Keep the original ResNet model for compatibility
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
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second block - increasing filters
    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third block
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Classification block
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

# Keep the other models for compatibility
def create_efficient_model(input_shape=(128, 128, 1), num_classes=8):
    # Original efficient model implementation
    # ... existing code ...
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
    # Original transformer hybrid implementation
    # ... existing code ...
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