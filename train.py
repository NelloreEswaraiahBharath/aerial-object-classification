# imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report
import numpy as np

# paths
train_path = "data/classification_dataset/train"
val_path = "data/classification_dataset/valid"

# data preprocessing
train = ImageDataGenerator(rescale=1./255)
val = ImageDataGenerator(rescale=1./255)

val_data = val.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False   # ⭐ IMPORTANT FIX
)
train_data = train.flow_from_directory(train_path, target_size=(224,224), batch_size=32, class_mode='binary')
val_data = val.flow_from_directory(val_path, target_size=(224,224), batch_size=32, class_mode='binary')

# cnn model
cnn = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(train_data, validation_data=val_data, epochs=2)

# transfer learning model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

tl_model = Model(inputs=base.input, outputs=output)
tl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tl_model.fit(train_data, validation_data=val_data, epochs=2)

# evaluation
preds = tl_model.predict(val_data)
pred_labels = (preds > 0.5).astype(int)
print(classification_report(val_data.classes, pred_labels))

# save best model
tl_model.save("models/best_model.h5")