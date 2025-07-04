import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import time
import psutil

dataset_path = kagglehub.dataset_download("gauravduttakiit/coffee-leaf-diseases")
print("Path to dataset files:", dataset_path)
# The CSV files train.csv and test.csv seem to describe the dataset but are not directly used by flow_from_directory, which infers classes from directory names.
# train = pd.read_csv("{}/{}".format(dataset_path, "train.csv"))
# test = pd.read_csv("{}/{}".format(dataset_path, "test.csv"))

if not os.path.exists(dataset_path):
    print(f"Error: Dataset directory not found at {dataset_path}. Please upload and extract the dataset.")
else:

    img_height, img_width = 255, 255
    batch_size = 2
    learning_rate = 0.001
    epochs = 200
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Assuming directory structure within the downloaded dataset path:
    # dataset_path/train/...
    # dataset_path/test/...
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
         print(f"Error: Train or validation directory not found. Please ensure your dataset is split into 'train' and 'test' subdirectories within {dataset_path}.")
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False) # Keep shuffle=False for consistent evaluation

        # Get the number of classes
        num_classes = len(train_generator.class_indices)
        print(f"Number of classes: {num_classes}")
        print(f"Class indices: {train_generator.class_indices}")

        # --- Build ResNet50 Model (with Regularization and Dropout) ---
        print("Building ResNet50 model...")
        base_model_50 = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        x = base_model_50.output
        x = GlobalAveragePooling2D()(x)
        predictions_50 = Dense(num_classes, activation='softmax')(x) # Corrected output units and activation
        model_resnet50 = Model(inputs=base_model_50.input, outputs=predictions_50)

        # Freeze the layers of the base model
        for layer in base_model_50.layers:
            layer.trainable = False

        # Compile the model
        model_resnet50.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        print("ResNet50 Summary: ")
        print(model_resnet50.summary())

         # --- Build ResNet101 Model (with Regularization and Dropout) ---
        # NOTE: The original code had a potential copy-paste error, using base_model_50.input for the ResNet101 model.
        # We will correct this to use base_model_101.input.
        print("Building ResNet101 model...")
        base_model_101 = ResNet101(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        x = base_model_101.output
        x = GlobalAveragePooling2D()(x)
        predictions_101 = Dense(num_classes, activation='softmax')(x) # Corrected output units and activation
        model_resnet101 = Model(inputs=base_model_101.input, outputs=predictions_101) # Corrected input

        # Freeze the layers of the base model
        for layer in base_model_101.layers:
            layer.trainable = False

        # Compile the model
        model_resnet101.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        print("ResNet101 Summary: ")
        print(model_resnet101.summary())

         # --- Train the models ---
        print("\nTraining ResNet50 model...")
        history_resnet50 = model_resnet50.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            #steps_per_epoch=200,
            epochs=epochs,
            #callbacks = [early_stopping],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            verbose = 1)

        print("\nTraining ResNet101 model...")
        history_resnet101 = model_resnet101.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            #steps_per_epoch=200,
            epochs=epochs,
            #callbacks = [early_stopping],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            verbose = 1)

        print("\nTraining complete.")

        # --- Save the models ---
        print("\nSaving the models...")
        model_resnet50.save('resnet50_coffee_leaf_disease.h5')
        model_resnet101.save('resnet101_coffee_leaf_disease.h5')

        # --- Evaluate performances ---
        print("\nEvaluating ResNet50 model...")

        # Get predictions for validation set
        validation_generator.reset() # Reset generator to ensure predictions are on the correct data order
        Y_pred_50 = model_resnet50.predict(validation_generator, steps=validation_generator.samples // batch_size + 1) # Use steps for full prediction
        y_pred_classes_50 = np.argmax(Y_pred_50, axis=1)
        y_true_50 = validation_generator.classes[:len(y_pred_classes_50)] # Slice true labels to match predictions

        # Calculate metrics
        accuracy_50 = accuracy_score(y_true_50, y_pred_classes_50)
        precision_50 = precision_score(y_true_50, y_pred_classes_50, average='weighted')
        f1_50 = f1_score(y_true_50, y_pred_classes_50, average='weighted')

        print(f"ResNet50 - Accuracy: {accuracy_50:.4f}")
        print(f"ResNet50 - Precision: {precision_50:.4f}")
        print(f"ResNet50 - F1-score: {f1_50:.4f}")
        print("ResNet50 - Classification Report:")
        # Ensure target_names match the order in the confusion matrix/report
        target_names = [k for k,v in sorted(validation_generator.class_indices.items(), key=lambda item: item[1])]
        print(classification_report(y_true_50, y_pred_classes_50, target_names=target_names))
        print("ResNet50 - Confusion Matrix:")
        print(confusion_matrix(y_true_50, y_pred_classes_50))

        print("\nEvaluating ResNet101 model...")
        validation_generator.reset() # Reset generator again
        Y_pred_101 = model_resnet101.predict(validation_generator, steps=validation_generator.samples // batch_size + 1) # Use steps for full prediction
        y_pred_classes_101 = np.argmax(Y_pred_101, axis=1)
        y_true_101 = validation_generator.classes[:len(y_pred_classes_101)] # Slice true labels to match predictions

        # Calculate metrics
        accuracy_101 = accuracy_score(y_true_101, y_pred_classes_101)
        precision_101 = precision_score(y_true_101, y_pred_classes_101, average='weighted')
        f1_101 = f1_score(y_true_101, y_pred_classes_101, average='weighted')

        print(f"ResNet101 - Accuracy: {accuracy_101:.4f}")
        print(f"ResNet101 - Precision: {precision_101:.4f}")
        print(f"ResNet101 - F1-score: {f1_101:.4f}")
        print("ResNet101 - Classification Report:")
        print(classification_report(y_true_101, y_pred_classes_101, target_names=target_names)) # Use the same target_names
        print("ResNet101 - Confusion Matrix:")
        print(confusion_matrix(y_true_101, y_pred_classes_101))

        # --- Show graphic of train loss and accuracy ---
        def plot_history(history, model_name):
            plt.figure(figsize=(12, 4))

            # Plot training & validation accuracy values
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title(f'{model_name} Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')

            # Plot training & validation loss values
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'{model_name} Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')

            plt.tight_layout()
            plt.show()

        print("\nPlotting training history...")
        plot_history(history_resnet50, 'ResNet50')
        plot_history(history_resnet101, 'ResNet101')

        # --- Compare resource usage during classification ---
        print("\nComparing resource usage for classification...")

        test_dir = os.path.join(dataset_path, 'test') # Assuming test data is in the 'test' subdir
        if not os.path.exists(test_dir):
            print(f"Error: Test directory not found at {test_dir}. Cannot compare resource usage on test data.")
        else:
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False) # Keep shuffle=False for consistent evaluation

            # Helper function to measure inference time and resource usage
            def measure_inference(model, generator, model_name):
                start_time = time.time()
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss # Resident Set Size (RAM usage)

                # Predict on the test set
                predictions = model.predict(generator, steps=generator.samples // batch_size + 1) # Use steps for full prediction

                end_time = time.time()
                final_memory = process.memory_info().rss
                inference_time = end_time - start_time
                memory_used = final_memory - initial_memory

                print(f"\n{model_name} Inference on Test Data:")
                print(f"  Inference Time: {inference_time:.4f} seconds")
                print(f"  Memory Used: {memory_used / (1024 * 1024):.2f} MB") # Convert bytes to MB

            # Measure for ResNet50
            test_generator.reset() # Reset generator before each prediction
            measure_inference(model_resnet50, test_generator, "ResNet50")

            # Measure for ResNet101
            test_generator.reset() # Reset generator before each prediction
            measure_inference(model_resnet101, test_generator, "ResNet101")

            print("\nComparison complete.")