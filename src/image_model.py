from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_image_model():
    base = EfficientNetB0(weights = 'imagenet', include_top = False)

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation = 'relu')(x)
    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base.input, outputs = output)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

# 👉 Model definition