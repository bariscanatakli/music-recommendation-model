import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data.data_loader import load_dataset
from models.advanced_models import (
    create_model, create_efficient_model, create_transformer_hybrid,
    create_ensemble_model, create_improved_model
)
from utils import setup_gpu_memory, plot_history, plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train music genre classification model')
    parser.add_argument('--model', type=str, default='improved',
                        choices=['improved', 'ensemble', 'resnet', 'efficient', 'transformer'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dataset_size', type=float, default=0.75, help='Proportion of data to use')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    return parser.parse_args()


def create_selected_model(model_name, input_shape, num_classes, lr=0.0001):
    """Create the selected model architecture."""
    print(f"Creating {model_name} model...")
    
    if model_name == "improved":
        model = create_improved_model(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "ensemble":
        model = create_ensemble_model(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "resnet":
        model = create_model(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "efficient":
        model = create_efficient_model(input_shape=input_shape, num_classes=num_classes)
    elif model_name == "transformer":
        model = create_transformer_hybrid(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
        
    return model


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup GPU memory growth
    setup_gpu_memory()
    
    # Create output directories
    model_dir = os.path.join(args.output_dir, 'models')
    plots_dir = os.path.join(args.output_dir, 'plots')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    for directory in [model_dir, plots_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load dataset
    train_x, train_y, test_x, test_y, n_classes, genre_names = load_dataset(
        verbose=1, mode="Train", datasetSize=args.dataset_size
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Training labels shape: {train_y.shape}")
    print(f"Testing data shape: {test_x.shape}")
    print(f"Testing labels shape: {test_y.shape}")
    print(f"Number of classes: {n_classes}")
    
    # Create model
    input_shape = (train_x.shape[1], train_x.shape[2], 1)
    model = create_selected_model(args.model, input_shape, n_classes, args.lr)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, f'best_{args.model}_model.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.000001
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    # Reshape data for training (add channel dimension)
    train_x_reshaped = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    test_x_reshaped = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
    
    # Fit data generator
    datagen.fit(train_x_reshaped)
    
    # Create training generator
    train_generator = datagen.flow(
        train_x_reshaped,
        train_y,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Train model
    print(f"\nStarting training {args.model} model for {args.epochs} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_x_reshaped) // args.batch_size,
        epochs=args.epochs,
        validation_data=(test_x_reshaped, test_y),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot and save training history
    history_path = os.path.join(plots_dir, f'{args.model}_training_history.png')
    plot_history(history, save_path=history_path)
    
    # Evaluate model
    print("\nEvaluating model on test data:")
    test_loss, test_acc = model.evaluate(test_x_reshaped, test_y, verbose=1)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save final model
    model.save(os.path.join(model_dir, f'music_genre_classifier_{args.model}.h5'))
    model.save(os.path.join(model_dir, f'music_genre_classifier_{args.model}.keras'))
    print(f"Model saved to {model_dir}")
    
    # Generate predictions for confusion matrix
    print("\nGenerating predictions for confusion matrix...")
    y_pred = model.predict(test_x_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(test_y, axis=1)
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=genre_names))
    
    # Plot confusion matrices
    conf_matrix_path = os.path.join(plots_dir, f'{args.model}_confusion_matrix.png')
    norm_conf_matrix_path = os.path.join(plots_dir, f'{args.model}_normalized_confusion_matrix.png')
    
    plot_confusion_matrix(
        y_true_classes, y_pred_classes, 
        genre_names, save_path=conf_matrix_path
    )
    
    plot_confusion_matrix(
        y_true_classes, y_pred_classes, 
        genre_names, save_path=norm_conf_matrix_path, 
        normalized=True
    )


if __name__ == "__main__":
    main()