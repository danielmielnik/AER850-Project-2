import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# %% Data Preprocessing
# Establishing Train, Validation, and Test Directories
train_dir = 'data/train'
validation_dir = 'data/valid'
test_dir = 'data/test'

# Performing data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
 
# Creating the train and validation generators
train_generator = image_dataset_from_directory(train_dir, image_size=(256, 256), batch_size=128, label_mode='categorical')
validation_generator = image_dataset_from_directory(validation_dir, image_size=(256, 256), batch_size=128, label_mode='categorical')

# %% Building Model 1 (Modified Sequential Model)
seq_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

seq_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
seq_model.summary()

# Training Model 1 with EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
history_1 = seq_model.fit(train_generator, validation_data=validation_generator, epochs=15, callbacks=[early_stopping, lr_scheduler])

seq_model.save('seq_model.keras')

# %% Building Model 2
# Convolution 2D and Max Pooling
model_2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(units=128, activation='elu'),
    Dropout(0.5),
    Dense(units=3, activation='softmax')
])

# Compiling Model 2
model_2.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model 2 Summary
model_2.summary()

# Traing Model 2
history_2 = model_2.fit(train_generator, validation_data=validation_generator, epochs=15)

# Saving Model 2
model_2.save('model_2.keras')

# %% Step 4 Model Evaluation

history_list = [history_1, history_2]
model_list = [seq_model, model_2]
i = 0

# For loop to evaluate each model
for history in history_list:
    # Model Evaluation
    test_loss, test_acc = model_list[i].evaluate(train_generator)
    print(f'Test accuracy: {test_acc:.4f}')
    i += 1
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model {i} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model {i} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
  
    plt.show()

