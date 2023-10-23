import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import tensorflow as tf
from tensorflow import keras

class BoundaryType(Enum):
    PERIODIC = 'periodic'

class ScalarField:
    def __init__(self, lattice_size, mass=1.0, lattice_spacing=0.1, dt=0.01, lambda_interaction=0.1, boundary=BoundaryType.PERIODIC):
        self.lattice_size = lattice_size
        self.lattice_spacing = lattice_spacing
        self.dt = dt
        self.mass = mass
        self.lambda_interaction = lambda_interaction
        self.field = np.random.rand(*lattice_size)
        self.boundary = boundary

    def _handle_boundary(self, array):
        if self.boundary == BoundaryType.PERIODIC:
            return np.pad(array, pad_width=1, mode='wrap')[1:-1, 1:-1]
        return array

    def _laplacian(self):
        laplacian = - len(self.lattice_size) * self.field
        for dim in range(len(self.lattice_size)):
            shift_positive = [slice(None)] * len(self.lattice_size)
            shift_negative = [slice(None)] * len(self.lattice_size)
            
            # Handling periodic boundaries:
            shift_positive[dim] = np.roll(self.field, -1, axis=dim)
            shift_negative[dim] = np.roll(self.field, 1, axis=dim)
            
            laplacian += shift_positive[dim] + shift_negative[dim]
        return laplacian


    def evolve(self, steps=10):
        field_prev = np.copy(self.field)
        for _ in range(steps):
            laplacian = self._laplacian()
            field_next = (2 * self.field - field_prev + (self.lattice_spacing ** 2 * self.dt ** 2) *
                          (-self.mass ** 2 * self.field + laplacian - self.lambda_interaction * self.field ** 3))
            field_prev, self.field = self.field, field_next
        return self.field

    def visualize(self, title=""):
        plt.imshow(self.field, cmap="viridis", origin="lower")
        plt.colorbar()
        plt.title(title)
        plt.show()

def generate_dataset(field, samples=1000, evolution_steps=10, noise_std=0.05):
    inputs, outputs = [], []
    for _ in range(samples):
        field.field = np.random.rand(*field.lattice_size)
        noisy_field = field.field + noise_std * np.random.randn(*field.lattice_size)
        inputs.append(noisy_field)
        outputs.append(field.evolve(steps=evolution_steps))
    return np.array(inputs), np.array(outputs)

def build_res_block(input_layer, filters):
    x = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(input_layer)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    return keras.layers.Add()([input_layer, x])

def build_model(input_shape, conv_layers=2, dense_layers=1, filters=[32, 64], dense_units=[256]):
    input_tensor = keras.layers.Input(shape=input_shape)
    x = keras.layers.Reshape((*input_shape, 1))(input_tensor)

    for i in range(conv_layers):
        x = keras.layers.Conv2D(filters[i], (3, 3), activation='relu', padding='same')(x)
        x = build_res_block(x, filters[i])
        x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Flatten()(x)
    for j in range(dense_layers):
        x = keras.layers.Dense(dense_units[j], activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(np.prod(input_shape))(x)
    x = keras.layers.Reshape(input_shape)(x)

    model = keras.models.Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

class TrainingConfig:
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 3
    BUFFER_SIZE = 10000  # For tf.data.Dataset shuffling

if __name__ == '__main__':
    lattice_size = (32, 32)
    field = ScalarField(lattice_size, lambda_interaction=0.1, boundary=BoundaryType.PERIODIC)

    X_train, y_train = generate_dataset(field, samples=5000, evolution_steps=20, noise_std=0.05)

    model = build_model(lattice_size, conv_layers=3, dense_layers=2, filters=[32, 64, 128], dense_units=[256, 128])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=TrainingConfig.PATIENCE, restore_best_weights=True)
    lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=TrainingConfig.EPOCHS, batch_size=TrainingConfig.BATCH_SIZE, validation_split=0.2, callbacks=[early_stop, lr_schedule, reduce_lr])

    # Visualization and testing the model can be added here as needed
