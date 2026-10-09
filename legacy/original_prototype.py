# -*- coding: utf-8 -*-
"""Fahad_MRI_Proj.ipynb
Muhammad Fahad

Contact for Original file is located at : fahadvine@gmail.com
"""

import os
import nibabel as nib #we are using this for .nii, .nii.gz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. Load Dataset from Google Drive
from google.colab import drive
drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/BraTs2019S/Training/HGG/BraTS19_2013_2_1/'
modalities = ['flair', 't1', 't1ce', 't2']

# Function to load modalities and segmentation
def load_patient(path):
    volume = []
    for modality in modalities:
        img = nib.load(os.path.join(path, f'{modality}.nii.gz')).get_fdata()
        volume.append(img)
    mask = nib.load(os.path.join(path, 'seg.nii.gz')).get_fdata()
    return np.stack(volume, axis=-1), mask

def load_brats_case(case_path):
    """Load one BraTS case with all modalities"""
    # Verify case directory exists
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case directory not found: {case_path}")

    # Get base filename (e.g., 'BraTS19_2013_10_1')
    case_id = os.path.basename(case_path)

    # Load each modality
    modalities = {
        'flair': f"{case_id}_flair.nii.gz",
        't1': f"{case_id}_t1.nii.gz",
        't1ce': f"{case_id}_t1ce.nii.gz",
        't2': f"{case_id}_t2.nii.gz",
        'seg': f"{case_id}_seg.nii.gz"  # Segmentation mask
    }

    data = {}
    for mod, fname in modalities.items():
        file_path = os.path.join(case_path, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data[mod] = nib.load(file_path).get_fdata()

    return data['t1ce'], data['seg']  # Return T1CE and mask

# Example usage
try:
    # Update this path to your specific case
    case_path = "/content/drive/MyDrive/Training/HGG/BraTS19_2013_10_1"

    # Load the data
    volume, mask = load_brats_case(case_path)
    print(f"Volume shape: {volume.shape}")  # Should be (240, 240, 155)
    print(f"Mask unique values: {np.unique(mask)}")  # Should be [0, 1, 2, 4]

    # Visualize middle slice
    slice_idx = volume.shape[-1] // 2

    plt.figure(figsize=(12, 6))

    # Display T1CE image
    plt.subplot(1, 2, 1)
    plt.imshow(volume[:, :, slice_idx], cmap='gray')
    plt.title('T1CE Image')
    plt.axis('off')

    # Display mask (with different colors for each class)
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, slice_idx], cmap='jet', vmin=0, vmax=4)
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {str(e)}")
    print("\nDebugging steps:")
    print(f"1. Verify this path exists: {case_path}")
    print(f"2. Check files in directory: !ls -l {case_path}")
    print("3. Confirm Google Drive is properly mounted")

def show_slices(volume, mask, slice_num=None):
    """Display MRI and segmentation slices with improved visualization"""
    # Auto-select middle slice if none specified
    if slice_num is None:
        slice_num = volume.shape[2] // 2

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # MRI Slice (assuming volume is 3D [H,W,D] from previous loading)
    axes[0].imshow(volume[:, :, slice_num], cmap='gray', interpolation='none')
    axes[0].set_title(f'MRI Slice (T1CE)\nSlice {slice_num}/{volume.shape[2]-1}')
    axes[0].axis('off')

    # Segmentation Mask
    seg_slice = mask[:, :, slice_num]
    im = axes[1].imshow(seg_slice, cmap='jet',
                       vmin=0, vmax=4,
                       interpolation='none')
    axes[1].set_title('Tumor Segmentation')
    axes[1].axis('off')

    # Add colorbar for segmentation classes
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 4])
    cbar.set_ticklabels(['Background', 'Necrotic', 'Edema', 'Enhancing'])

    plt.tight_layout()
    plt.show()

# Example usage with auto-middle-slice selection
show_slices(volume, mask)  # Will show middle slice automatically

# Or specify a slice number
# show_slices(volume, mask, slice_num=70)

def show_slices(volume, mask, slice_num=None):
    """
    Display MRI and segmentation slices with enhanced visualization
    Args:
        volume: 3D numpy array (H, W, D) from T1CE modality
        mask: 3D numpy array (H, W, D) with segmentation labels
        slice_num: Optional slice number to display (defaults to middle slice)
    """
    # Auto-calculate middle slice if not specified
    if slice_num is None:
        slice_num = volume.shape[-1] // 2

    # Verify slice number is valid
    assert 0 <= slice_num < volume.shape[-1], f"Invalid slice number {slice_num} for volume with depth {volume.shape[-1]}"

    # Create figure with adjusted subplot spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0.1})

    # Display MRI slice (T1CE)
    ax1.imshow(volume[:, :, slice_num], cmap='gray', interpolation='none')
    ax1.set_title(f'T1CE Slice {slice_num}', fontsize=12, pad=10)
    ax1.axis('off')

    # Display segmentation with custom colormap
    cmap = plt.cm.jet
    cmap.set_under('black', alpha=0)
    seg_slice = mask[:, :, slice_num]

    im = ax2.imshow(seg_slice, cmap=cmap,
                   vmin=0.5, vmax=4,  # Adjusted to better show tumor regions
                   interpolation='none')
    ax2.set_title('Tumor Segmentation', fontsize=12, pad=10)
    ax2.axis('off')

    # Add professional colorbar
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_ticks([1, 2, 4])
    cbar.set_ticklabels([
        'Necrotic Core (1)',
        'Edema (2)',
        'Enhancing Tumor (4)'
    ])
    cbar.ax.tick_params(labelsize=10)

    # Add slice navigator text
    plt.suptitle(f'BraTS19 Slice Visualization - Depth: {slice_num}/{volume.shape[-1]-1}',
                 y=0.95, fontsize=14)

    plt.tight_layout()
    plt.show()

# Example usage with your data:
try:
    # Load your case first (using our previous loading function)
    case_path = "/content/drive/MyDrive/Training/HGG/BraTS19_2013_10_1"
    volume, mask = load_brats_case(case_path)  # From previous solution

    # Visualize with different options:
    show_slices(volume, mask)  # Middle slice
    show_slices(volume, mask, 70)  # Specific slice
    show_slices(volume, mask, volume.shape[-1]-10)  # Near-end slice

except Exception as e:
    print(f"Visualization error: {str(e)}")

import numpy as np
from skimage.transform import resize

def preprocess_volume(volume, target_shape=(128, 128, 128)):
    """
    Full preprocessing pipeline for BraTS volumes:
    1. Normalization (zero-mean, unit-variance)
    2. Resizing (optional)
    3. Channel addition (for CNN compatibility)
    """
    # 1. Normalization (zero-mean, unit-variance)
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)  # Added epsilon for stability

    # 2. Resize if target shape is different
    if volume.shape != target_shape:
        volume = resize(volume, target_shape, order=1, mode='constant', preserve_range=True)

    # 3. Add channel dimension (for PyTorch compatibility)
    volume = np.expand_dims(volume, axis=0)  # Shape becomes (1, H, W, D)

    return volume

def preprocess_mask(mask, target_shape=(128, 128, 128)):
    """Preprocess segmentation mask (nearest-neighbor resizing)"""
    if mask.shape != target_shape:
        mask = resize(mask, target_shape, order=0, mode='constant', preserve_range=True)
    return mask.astype(np.uint8)

# Example usage with our previous data loading:
try:
    # Load raw data
    case_path = "/content/drive/MyDrive/Training/HGG/BraTS19_2013_10_1"
    raw_volume, raw_mask = load_brats_case(case_path)

    # Preprocess
    processed_volume = preprocess_volume(raw_volume)
    processed_mask = preprocess_mask(raw_mask)

    print(f"Original shape: {raw_volume.shape} -> Processed shape: {processed_volume.shape}")
    print(f"Value range: {np.min(processed_volume):.2f} to {np.max(processed_volume):.2f}")

    # Visualize before/after
    print("\nOriginal slice:")
    show_slices(raw_volume, raw_mask, 70)

    print("\nProcessed slice:")
    show_slices(processed_volume[0], processed_mask, 50)  # Note: [0] to remove channel dim for visualization

except Exception as e:
    print(f"Processing error: {str(e)}")

# Full dataset preprocessing (example)
def preprocess_dataset(volumes, masks, target_shape=(128, 128, 128)):
    """Batch process entire dataset"""
    X = np.array([preprocess_volume(v, target_shape) for v in volumes])
    y = np.array([preprocess_mask(m, target_shape) for m in masks])
    return X, y

# Usage:
# X_train, y_train = preprocess_dataset(raw_train_volumes, raw_train_masks)
# X_val, y_val = preprocess_dataset(raw_val_volumes, raw_val_masks)

from tensorflow.keras import layers, models
import tensorflow as tf

def conv_block(x, filters, dropout_rate=0.2):
    """Enhanced convolutional block with batch norm and dropout"""
    x = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(x, filters):
    """Encoder block with skip connection"""
    f = conv_block(x, filters)
    p = layers.MaxPool3D(2)(f)
    return f, p

def decoder_block(x, skip, filters):
    """Robust decoder block with automatic shape matching"""
    # Upsample with Conv3DTranspose
    x = layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)

    # Match channels if needed
    if skip.shape[-1] != filters:
        skip = layers.Conv3D(filters, 1, padding='same')(skip)

    # Resize if dimensions don't match (alternative to cropping)
    if x.shape[1] != skip.shape[1]:
        target_size = skip.shape[1:4]
        x = layers.Resizing(*target_size)(x)

    x = layers.Concatenate()([x, skip])
    return conv_block(x, filters)

def build_brats_unet(input_shape=(128, 128, 128, 1)):
    """
    Optimized 3D U-Net for BraTS segmentation with:
    - Proper shape handling throughout network
    - Multi-class output (4 tumor classes)
    - Enhanced skip connections
    """
    inputs = layers.Input(input_shape)

    # Encoder path
    f1, p1 = encoder_block(inputs, 32)
    f2, p2 = encoder_block(p1, 64)
    f3, p3 = encoder_block(p2, 128)
    f4, p4 = encoder_block(p3, 256)

    # Bottleneck
    bottleneck = conv_block(p4, 512)

    # Decoder path
    d4 = decoder_block(bottleneck, f4, 256)
    d3 = decoder_block(d4, f3, 128)
    d2 = decoder_block(d3, f2, 64)
    d1 = decoder_block(d2, f1, 32)

    # Output layer (4 classes for BraTS)
    outputs = layers.Conv3D(4, 1, activation='softmax')(d1)

    model = models.Model(inputs, outputs, name='3D_UNet_BraTS')
    return model

# Training setup with proper initialization
def initialize_model():
    try:
        # Build model
        model = build_brats_unet(input_shape=(128, 128, 128, 1))

        # Custom loss function for class imbalance
        def brats_loss(y_true, y_pred):
            class_weights = tf.constant([0.1, 0.3, 0.3, 0.3])  # Background, Necrotic, Edema, Enhancing
            y_true = tf.cast(y_true, tf.int32)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(ce * tf.gather(class_weights, y_true))

        # Compile with appropriate metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=brats_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.MeanIoU(num_classes=4, name='mean_iou')
            ]
        )

        model.summary()
        return model

    except Exception as e:
        print(f"Model initialization error: {str(e)}")
        return None

# Example usage:
# model = initialize_model()
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#                    epochs=100, batch_size=2)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

def compile_and_train(X_train, y_train, X_val, y_val, input_shape=(128, 128, 128, 1)):
    """
    Complete training pipeline for BraTS segmentation with:
    1. Proper model initialization
    2. Multi-class loss function
    3. Optimized training parameters
    4. Enhanced callbacks
    """
    try:
        # 1. Model initialization
        model = build_brats_unet(input_shape)

        # 2. Custom BraTS loss function (handles class imbalance)
        def brats_loss(y_true, y_pred):
            # Class weights: Background:0.1, Necrotic:0.3, Edema:0.3, Enhancing:0.3
            weights = tf.constant([0.1, 0.3, 0.3, 0.3])
            y_true = tf.cast(y_true, tf.int32)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(ce * tf.gather(weights, y_true))

        # 3. Metrics for tumor subregions
        metrics = [
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=4),
            tf.keras.metrics.SparseCategoricalAccuracy()
        ]

        # 4. Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=brats_loss,
            metrics=metrics
        )

        # 5. Training callbacks
        callbacks = [
            ModelCheckpoint('best_model.h5',
                          monitor='val_loss',
                          save_best_only=True,
                          mode='min'),
            ReduceLROnPlateau(monitor='val_loss',
                             factor=0.1,
                             patience=5,
                             verbose=1),
            EarlyStopping(monitor='val_loss',
                         patience=10,
                         restore_best_weights=True)
        ]

        # 6. Train model
        history = model.fit(
            X_train,  # Shape: (N, 128, 128, 128, 1)
            y_train,  # Shape: (N, 128, 128, 128) with values 0-3
            batch_size=2,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

        return model, history

    except Exception as e:
        print(f"Training error: {str(e)}")
        return None, None

# Usage example:
# model, history = compile_and_train(
#     X_train, y_train,
#     X_val, y_val,
#     input_shape=X_train[0].shape
# )

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np

def compile_and_train(X_train, y_train, X_val, y_val):
    """
    Complete BraTS training pipeline with proper mask reshaping:
    1. Converts masks to categorical format
    2. Uses multi-class loss function
    3. Includes BraTS-specific metrics
    """
    try:
        # 1. Prepare masks (convert from 0,1,2,4 to 0,1,2,3 and make categorical)
        y_train_cat = tf.keras.utils.to_categorical(
            np.where(y_train == 4, 3, y_train),  # Convert 4→3
            num_classes=4
        )
        y_val_cat = tf.keras.utils.to_categorical(
            np.where(y_val == 4, 3, y_val),
            num_classes=4
        )

        # 2. Build model (from our previous implementation)
        model = build_brats_unet(input_shape=X_train[0].shape)

        # 3. BraTS-optimized loss function
        def brats_loss(y_true, y_pred):
            # Class weights: Background, Necrotic, Edema, Enhancing
            weights = tf.constant([0.1, 0.3, 0.3, 0.3])
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(ce * tf.reduce_sum(weights * y_true, axis=-1))

        # 4. Tumor-specific metrics
        def dice_coef(y_true, y_pred, class_idx):
            y_true = y_true[..., class_idx]
            y_pred = y_pred[..., class_idx]
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

        metrics = [
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=4),
            lambda yt, yp: dice_coef(yt, yp, 1),  # Necrotic
            lambda yt, yp: dice_coef(yt, yp, 2),  # Edema
            lambda yt, yp: dice_coef(yt, yp, 3)   # Enhancing
        ]

        # 5. Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=brats_loss,
            metrics=metrics
        )

        # 6. Callbacks
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.1, patience=5),
            EarlyStopping(patience=10, restore_best_weights=True)
        ]

        # 7. Train
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=2,
            callbacks=callbacks
        )

        return model, history

    except Exception as e:
        print(f"Training error: {str(e)}")
        return None, None

# Usage example:
# After preprocessing with our previous functions:
# X_train, y_train = preprocess_dataset(train_volumes, train_masks)
# X_val, y_val = preprocess_dataset(val_volumes, val_masks)

# Then train:
# model, history = compile_and_train(X_train, y_train, X_val, y_val)

import os
import tensorflow as tf   #Build, train, and deploy machine learning models.
from datetime import datetime  #

def save_brats_model(model, history=None, base_path="/content/drive/MyDrive/BraTS_Models"):
    """
    Enhanced model saving with:
    1. Versioned saves (timestamps)
    2. Full model + weights options
    3. Metadata preservation
    4. Training history saving
    """
    try:
        # Create save directory if needed
        os.makedirs(base_path, exist_ok=True)

        # Generate versioned filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"3d_unet_brats19_{timestamp}"

        # 1. Save full HDF5 model (architecture + weights + optimizer)
        model_path = os.path.join(base_path, f"{model_name}.h5")
        model.save(model_path)
        print(f"Full model saved to {model_path}")

        # 2. Save weights only (smaller file)
        weights_path = os.path.join(base_path, f"{model_name}_weights.h5")
        model.save_weights(weights_path)
        print(f"Weights saved to {weights_path}")

        # 3. Save training history if available
        if history is not None:
            history_path = os.path.join(base_path, f"{model_name}_history.pkl")
            import pickle
            with open(history_path, 'wb') as f:
                pickle.dump(history.history, f)
            print(f"Training history saved to {history_path}")

        # 4. Save metadata
        metadata = {
            'input_shape': model.input_shape[1:],
            'output_classes': model.output_shape[-1],
            'training_date': timestamp,
            'framework': f"TensorFlow {tf.__version__}"
        }
        metadata_path = os.path.join(base_path, f"{model_name}_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

        return model_path

    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return None

# Usage example (after training):
# model_path = save_brats_model(model, history)
# print(f"Model saved at: {model_path}")
