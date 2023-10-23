import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class ScalarField:
    def __init__(self, lattice_size, mass=1.0, lattice_spacing=0.1, dt=0.01, lambda_interaction=0.1, boundary='periodic'):
        self.lattice_size = lattice_size
        self.lattice_spacing = lattice_spacing
        self.dt = dt
        self.mass = mass
        self.lambda_interaction = lambda_interaction
        self.field = np.random.rand(*lattice_size)
        self.boundary = boundary

    def _laplacian(self):
        laplacian = - len(self.lattice_size) * self.field
        for dim in range(len(self.lattice_size)):
            shift_positive = [slice(None)] * len(self.lattice_size)
            shift_positive[dim] = slice(1, None)
            shift_negative = [slice(None)] * len(self.lattice_size)
            shift_negative[dim] = slice(None, -1)
            laplacian[tuple(shift_positive)] += self.field[tuple(shift_negative)]
            laplacian[tuple(shift_negative)] += self.field[tuple(shift_positive)]
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

def build_model(input_shape):
    input_tensor = keras.layers.Input(shape=input_shape)
    x = keras.layers.Reshape((*input_shape, 1))(input_tensor)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = build_res_block(x, 32)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = build_res_block(x, 64)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(np.prod(input_shape))(x)
    x = keras.layers.Reshape(input_shape)(x)
    model = keras.models.Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    return model

class FieldEvolutionPredictor:
    def __init__(self, model, field_class):
        self.model = model
        self.field_class = field_class

    def predict_evolution(self, initial_field, training_steps=1):
        predicted_evolution = self.model.predict(initial_field.reshape(1, *self.field_class.lattice_size))[0]
        self.field_class.field = initial_field
        actual_evolution = self.field_class.evolve(steps=20)
        X_new, y_new = initial_field.reshape(1, *self.field_class.lattice_size), actual_evolution.reshape(1, *self.field_class.lattice_size)
        self.model.fit(X_new, y_new, epochs=training_steps, verbose=0)
        return predicted_evolution

if __name__ == '__main__':
    lattice_size = (32, 32)
    field = ScalarField(lattice_size, lambda_interaction=0.1, boundary='periodic')

    X_train, y_train = generate_dataset(field, samples=5000, evolution_steps=20, noise_std=0.05)

    model = build_model(lattice_size)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop, lr_schedule, reduce_lr])

    predictor = FieldEvolutionPredictor(model, field)
    new_field = np.random.rand(*lattice_size)
    predicted_evolution = predictor.predict_evolution(new_field)
    field.visualize(predicted_evolution, "Predicted Evolution using Neural Network")

    field.field = new_field
    actual_evolution = field.evolve(steps=20)
    field.visualize(actual_evolution, "Actual Evolution")
