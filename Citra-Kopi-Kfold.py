!pip install scikit-learn
!pip install keras-tuner
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import os
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import time
import psutil
import keras_tuner as kt

# Ensure TensorFlow is installed (redundant with the import but good practice)
!pip install tensorflow

# Download dataset
dataset_path = kagglehub.dataset_download("gauravduttakiit/coffee-leaf-diseases")
print("Path to dataset files:", dataset_path)

# Check if the dataset directory exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset directory not found at {dataset_path}. Please ensure the dataset is downloaded correctly.")
else:
    img_height, img_width = 224, 224
    batch_size = 32 # Reverting to a more standard batch size for tuning
    epochs = 50 # Reduce epochs for tuning speed

    # Assuming directory structure within the downloaded dataset path:
    # dataset_path/train/...
    # dataset_path/test/...
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test') # Use test_dir for the final evaluation

    # Check if train and test directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
         print(f"Error: Train or test directory not found. Please ensure your dataset is split into 'train' and 'test' subdirectories within {dataset_path}.")
    else:
        # We need to load all images for K-Fold Cross Validation
        # flow_from_directory with a single directory will handle this
        # We'll use the 'train' directory for K-Fold splitting.
        # The 'test' directory will be used for final evaluation after tuning.

        all_datagen = ImageDataGenerator(rescale=1./255) # Rescale for initial loading
        all_generator = all_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False) # Keep shuffle=False to maintain order for splitting

        # Get image paths and labels from the generator
        image_files = [os.path.join(train_dir, img_path) for img_path in all_generator.filenames]
        labels = all_generator.classes
        class_indices = all_generator.class_indices
        num_classes = len(class_indices)
        print(f"Number of classes: {num_classes}")
        print(f"Class indices: {class_indices}")

        # Create a DataFrame for easier splitting
        data_df = pd.DataFrame({'filename': image_files, 'label': labels})
        data_df['label_name'] = data_df['label'].map({v: k for k, v in class_indices.items()}) # Add label names

        # --- K-Fold Cross Validation Setup ---
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # --- Hyperparameter Tuning with Keras Tuner ---
        # We will perform tuning on the ResNet50 model as an example.
        # ResNet101 tuning would follow a similar structure.

        def build_resnet_model(hp):
            """Builds a ResNet50 model for hyperparameter tuning."""
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            # Tune the number of units in the dense layer
            hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
            x = Dense(units=hp_units, activation='relu', kernel_regularizer=l2(0.0001))(x)

            # Tune the dropout rate
            hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
            x = Dropout(hp_dropout)(x)

            predictions = Dense(num_classes, activation='softmax')(x) # Softmax for multi-class

            model = Model(inputs=base_model.input, outputs=predictions)

            # Tune the learning rate for the optimizer
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
            optimizer = Adam(learning_rate=hp_learning_rate)

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        # Setup the tuner
        tuner = kt.Hyperband(
            build_resnet_model,
            objective='val_accuracy',
            max_epochs=epochs,
            factor=3,
            directory='keras_tuner_dir',
            project_name='resnet50_tuning')

        # Prepare data for tuning (using a portion of the training data)
        # For simplicity and demonstration, we'll use 80% of the training data for tuning
        # and reserve 20% for validation during tuning.
        # In a real scenario, you might use a dedicated tuning set or nested cross-validation.
        from sklearn.model_selection import train_test_split
        train_tuning_df, val_tuning_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])

        train_tuning_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')

        val_tuning_datagen = ImageDataGenerator(rescale=1./255)

        train_tuning_generator = train_tuning_datagen.flow_from_dataframe(
            train_tuning_df,
            x_col='filename',
            y_col='label_name', # Use label_name string column for flow_from_dataframe
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        val_tuning_generator = val_tuning_datagen.flow_from_dataframe(
            val_tuning_df,
            x_col='filename',
            y_col='label_name',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        # Run the tuner search
        print("\nRunning Keras Tuner for ResNet50 hyperparameters...")
        tuner.search(train_tuning_generator,
                     epochs=epochs,
                     validation_data=val_tuning_generator,
                     steps_per_epoch=train_tuning_generator.samples // batch_size,
                     validation_steps=val_tuning_generator.samples // batch_size)

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"\nBest hyperparameters found for ResNet50:")
        print(f"  Learning Rate: {best_hps.get('learning_rate'):.4f}")
        print(f"  Dense Units: {best_hps.get('units')}")
        print(f"  Dropout Rate: {best_hps.get('dropout'):.2f}")

        # Build the best model using the optimal hyperparameters
        best_resnet50_model = tuner.hypermodel.build(best_hps)

        # --- K-Fold Cross Validation Training with Best Hyperparameters ---
        fold_results = []
        fold = 1

        print(f"\nStarting {n_splits}-Fold Cross Validation for ResNet50 with best hyperparameters...")

        for train_index, val_index in kf.split(data_df):
            print(f"\n--- Fold {fold}/{n_splits} ---")

            # Split data for the current fold
            train_fold_df = data_df.iloc[train_index]
            val_fold_df = data_df.iloc[val_index]

            # Data generators for the current fold
            train_fold_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode='nearest')

            val_fold_datagen = ImageDataGenerator(rescale=1./255)

            train_fold_generator = train_fold_datagen.flow_from_dataframe(
                train_fold_df,
                x_col='filename',
                y_col='label_name',
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

            val_fold_generator = val_fold_datagen.flow_from_dataframe(
                val_fold_df,
                x_col='filename',
                y_col='label_name',
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False) # Keep shuffle=False for consistent evaluation

            # Rebuild and compile the model for each fold to ensure fresh state
            # We use the best hyperparameters found earlier
            model = build_resnet_model(best_hps) # Use the function with best_hps

            # Train the model for the current fold
            history = model.fit(
                train_fold_generator,
                steps_per_epoch=train_fold_generator.samples // batch_size,
                epochs=epochs, # Use the reduced epochs for each fold
                validation_data=val_fold_generator,
                validation_steps=val_fold_generator.samples // batch_size,
                verbose=0) # Set verbose to 0 for cleaner output during folds

            # Evaluate the model on the validation data for this fold
            val_fold_generator.reset()
            loss, accuracy = model.evaluate(val_fold_generator, steps=val_fold_generator.samples // batch_size + 1, verbose=0)

            print(f"  Fold {fold} - Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
            fold_results.append(accuracy)

            fold += 1

        # Report average performance across folds
        avg_accuracy = np.mean(fold_results)
        std_accuracy = np.std(fold_results)
        print(f"\nAverage K-Fold Validation Accuracy (ResNet50): {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # --- Final Training on Full Training Data with Best Hyperparameters ---
        print("\nTraining the final ResNet50 model on the full training dataset with best hyperparameters...")

        # Use the original train_datagen and train_generator for the full training
        train_datagen_full = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')

        # Load all training data using flow_from_directory pointing to train_dir
        train_generator_full = train_datagen_full.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        # Rebuild the best model
        final_resnet50_model = build_resnet_model(best_hps)

        # Train the final model
        # Consider using more epochs for the final training after tuning
        final_epochs = 100 # Increase epochs for final training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Add early stopping

        # Prepare the validation generator for the final training (using the test data for evaluation)
        final_val_datagen = ImageDataGenerator(rescale=1./255)
        final_val_generator = final_val_datagen.flow_from_directory(
            test_dir, # Use the test data for final validation/evaluation
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        history_final_resnet50 = final_resnet50_model.fit(
            train_generator_full,
            steps_per_epoch=train_generator_full.samples // batch_size,
            epochs=final_epochs,
            callbacks=[early_stopping],
            validation_data=final_val_generator,
            validation_steps=final_val_generator.samples // batch_size)

        print("\nFinal ResNet50 training complete.")

        # --- Evaluate Final ResNet50 Performance on Test Data ---
        print("\nEvaluating Final ResNet50 model on Test Data...")
        final_val_generator.reset()
        Y_pred_final_50 = final_resnet50_model.predict(final_val_generator, steps=final_val_generator.samples // batch_size + 1)
        y_pred_classes_final_50 = np.argmax(Y_pred_final_50, axis=1)
        y_true_final_50 = final_val_generator.classes[:len(y_pred_classes_final_50)] # Slice true labels

        accuracy_final_50 = accuracy_score(y_true_final_50, y_pred_classes_final_50)
        precision_final_50 = precision_score(y_true_final_50, y_pred_classes_final_50, average='weighted')
        f1_final_50 = f1_score(y_true_final_50, y_pred_classes_final_50, average='weighted')

        print(f"\nFinal ResNet50 (on Test Data) - Accuracy: {accuracy_final_50:.4f}")
        print(f"Final ResNet50 (on Test Data) - Precision: {precision_final_50:.4f}")
        print(f"Final ResNet50 (on Test Data) - F1-score: {f1_final_50:.4f}")
        print("Final ResNet50 (on Test Data) - Classification Report:")
        target_names_final = [k for k,v in sorted(final_val_generator.class_indices.items(), key=lambda item: item[1])]
        print(classification_report(y_true_final_50, y_pred_classes_final_50, target_names=target_names_final))
        print("Final ResNet50 (on Test Data) - Confusion Matrix:")
        print(confusion_matrix(y_true_final_50, y_pred_classes_final_50))

        # --- Save the Final ResNet50 Model ---
        print("\nSaving the Final ResNet50 model...")
        final_resnet50_model.save('final_resnet50_coffee_leaf_disease.h5')

        # --- Plotting the final training history ---
        plot_history(history_final_resnet50, 'Final ResNet50')

        # --- Resource Usage Comparison (using the Final Model) ---
        print("\nComparing resource usage for classification with the Final ResNet50 model...")

        test_datagen_eval = ImageDataGenerator(rescale=1./255)
        test_generator_eval = test_datagen_eval.flow_from_directory(
            test_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        measure_inference(final_resnet50_model, test_generator_eval, "Final ResNet50")

        print("\nProcess completed.")

        # Note: ResNet101 tuning and K-Fold would follow a very similar process
        # to the ResNet50 steps outlined above. You would define a separate
        # build_resnet101_model function and run a separate tuner search.
