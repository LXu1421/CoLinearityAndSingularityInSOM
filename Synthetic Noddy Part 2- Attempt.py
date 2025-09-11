import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the same lithology mappings for consistency
lithology_mappings = {
    'Q004': {
        1: 'Mafic volcanic',
        2: 'Sedimentary cover',
        3: 'Psammitic sediment',
        4: 'Felsic intrusive',
        5: 'Pelitic sediment'
    },
    'Q002': {
        1: 'Metamorphic rocks',
        2: 'Sedimentary cover',
        3: 'Psammitic sediment',
        4: 'Pelitic sediment'
    },
    'Q001': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q003': {
        1: 'Sedimentary cover',
        2: 'Psammitic sediment',
        3: 'Felsic intrusive',
        4: 'Pelitic sediment'
    },
    'Q006': {
        1: 'Chert',
        2: 'Carbonaceous rock',
        3: 'Sedimentary cover',
        4: 'Psammitic sediment',
        5: 'Felsic intrusive',
        6: 'Pelitic sediment'
    },
    'Q005': {
        1: 'Intermediate intrusive',
        2: 'Chert',
        3: 'Sedimentary cover',
        4: 'Carbonaceous rock',
        5: 'Pelitic sediment',
        6: 'Psammitic sediment'
    }
}


def load_and_prepare_data(data_dir="SyntheticNoddy/Data"):
    """Load and prepare the data from NPZ files for training."""
    X_data = []
    y_data = []
    model_info = []

    for i in range(1, 7):
        q_key = f"Q{i:03d}"

        # Load geophysics data
        grav_data = np.load(os.path.join(data_dir, f"{q_key}-Grav_processed.npz"))
        mag_data = np.load(os.path.join(data_dir, f"{q_key}-Mag_processed.npz"))

        # Load planview data
        pv_data = np.load(os.path.join(data_dir, f"{q_key}-planview_processed.npz"))

        # Get the expanded planview (lithology labels)
        lithology = pv_data['expanded']

        # Convert RGB to single channel if needed
        if lithology.ndim == 3:
            lithology = (lithology[..., 0].astype(np.uint32) << 16 |
                         lithology[..., 1].astype(np.uint32) << 8 |
                         lithology[..., 2].astype(np.uint32))

        # Prepare features (8 channels: 4 from gravity, 4 from magnetic)
        grav_features = np.stack([grav_data['raw'], grav_data['1VD'],
                                  grav_data['tile'], grav_data['analytical_signal']], axis=-1)
        mag_features = np.stack([mag_data['raw'], mag_data['1VD'],
                                 mag_data['tile'], mag_data['analytical_signal']], axis=-1)

        # Combine all features
        features = np.concatenate([grav_features, mag_features], axis=-1)

        # Flatten for training
        X_flat = features.reshape(-1, features.shape[-1])
        y_flat = lithology.flatten()

        # Remove any NaN values
        valid_mask = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(y_flat)
        X_flat = X_flat[valid_mask]
        y_flat = y_flat[valid_mask]

        X_data.append(X_flat)
        y_data.append(y_flat)

        # Store model information for later analysis
        model_info.extend([(q_key, lithology_mappings[q_key].get(int(val), "Unknown"))
                           for val in y_flat])

    # Combine all data
    X = np.concatenate(X_data, axis=0)
    y = np.concatenate(y_data, axis=0)

    return X, y, model_info


def create_unet_model(input_shape, num_classes):
    """Create a U-Net model for semantic segmentation."""
    inputs = keras.Input(shape=input_shape)

    # Downsample path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bridge
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Upsample path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_cnn_model(input_shape, num_classes):
    """Create a simpler CNN model for classification."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(X, y, model_type='cnn', test_size=0.2, random_state=42):
    """Train a model on the prepared data."""
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and compile model
    if model_type == 'cnn':
        model = create_cnn_model((X_train.shape[1],), num_classes)
    elif model_type == 'unet':
        # For U-Net, we need to reshape the data to 2D
        # This is a simplified approach - in practice, you'd want to keep the spatial structure
        print("U-Net requires spatial data structure. Using CNN instead.")
        model = create_cnn_model((X_train.shape[1],), num_classes)
    else:
        raise ValueError("Model type must be 'cnn' or 'unet'")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=128,
        verbose=1
    )

    return model, history, le, scaler, X_test_scaled, y_test


def evaluate_model(model, history, X_test, y_test, le, scaler, output_dir="NN_Results"):
    """Evaluate the trained model and save results."""
    os.makedirs(output_dir, exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate classification report
    class_names = le.classes_
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close()

    # Save model
    model.save(os.path.join(output_dir, "lithology_model.h5"))

    # Save label encoder and scaler
    np.save(os.path.join(output_dir, "label_encoder_classes.npy"), le.classes_)
    np.save(os.path.join(output_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(output_dir, "scaler_scale.npy"), scaler.scale_)

    return report_df


def analyze_feature_importance(model, feature_names, output_dir="NN_Results"):
    """Analyze and visualize feature importance."""
    # Get weights from the first layer
    weights = model.layers[0].get_weights()[0]

    # Calculate feature importance (mean absolute weight)
    importance = np.mean(np.abs(weights), axis=1)

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()

    # Save to CSV
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    return importance_df


def main():
    print("Loading and preparing data...")
    X, y, model_info = load_and_prepare_data()

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Number of unique classes: {len(np.unique(y))}")

    # Define feature names for interpretation
    feature_names = [
        'Grav_raw', 'Grav_1VD', 'Grav_tile', 'Grav_analytical_signal',
        'Mag_raw', 'Mag_1VD', 'Mag_tile', 'Mag_analytical_signal'
    ]

    print("Training model...")
    model, history, le, scaler, X_test, y_test = train_model(X, y, model_type='cnn')

    print("Evaluating model...")
    report_df = evaluate_model(model, history, X_test, y_test, le, scaler)

    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(model, feature_names)

    print("Training and evaluation complete!")
    print("\nClassification Report:")
    print(report_df)
    print("\nTop 5 Most Important Features:")
    print(importance_df.head())

    # Save final summary
    summary = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(le.classes_),
        'test_accuracy': history.history['val_accuracy'][-1],
        'top_features': importance_df['Feature'].head(3).tolist()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("NN_Results/training_summary.csv", index=False)


if __name__ == "__main__":
    main()