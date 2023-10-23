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
        self.field_prev = np.copy(self.field)
        self.boundary = boundary

    def evolve(self, steps=10):
        """Evolve the scalar field."""
        for _ in range(steps):
            if self.boundary == 'periodic':
                laplacian = (np.roll(self.field, shift=1, axis=0) + np.roll(self.field, shift=-1, axis=0)
                             + np.roll(self.field, shift=1, axis=1) + np.roll(self.field, shift=-1, axis=1)
                             - 4 * self.field)
            elif self.boundary == 'dirichlet':
                laplacian = -4 * self.field
                laplacian[:-1] += self.field[1:]
                laplacian[1:] += self.field[:-1]
                laplacian[:, :-1] += self.field[:, 1:]
                laplacian[:, 1:] += self.field[:, :-1]
            
            field_next = (2 * self.field - self.field_prev + (self.lattice_spacing ** 2 * self.dt ** 2) *
                          (-self.mass ** 2 * self.field + laplacian - self.lambda_interaction * self.field ** 3))
            self.field_prev = np.copy(self.field)
            self.field = field_next
        return self.field

    def visualize(self, data, title=""):
        plt.imshow(data, cmap="viridis", origin="lower")
        plt.colorbar()
        plt.title(title)
        plt.show()

def generate_dataset(field, samples=1000, evolution_steps=10, noise_std=0.05):
    """Generate dataset of initial and evolved scalar field configurations with optional noise."""
    inputs = []
    outputs = []

    for _ in range(samples):
        field.field = np.random.rand(*field.lattice_size)
        noisy_field = field.field + noise_std * np.random.randn(*field.lattice_size)
        inputs.append(noisy_field)

        evolved_field = field.evolve(steps=evolution_steps)
        outputs.append(evolved_field)

    return np.array(inputs), np.array(outputs)

def build_model(input_shape):
    """Build an enhanced Convolutional Neural Network model."""
    model = keras.Sequential([
        keras.layers.Reshape((*input_shape, 1), input_shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(np.prod(input_shape)),
        keras.layers.Reshape(input_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class FieldEvolutionPredictor:
    def __init__(self, model, lattice_size, field_class):
        self.model = model
        self.lattice_size = lattice_size
        self.field_class = field_class

    def predict_evolution(self, initial_field, training_steps=1):
        # Model makes its prediction
        predicted_evolution = self.model.predict(initial_field.reshape(1, *self.lattice_size))[0]
        
        # Evolve the field using the ScalarField class
        self.field_class.field = initial_field
        actual_evolution = self.field_class.evolve(steps=20)
        
        # Train the model using this new data
        X_new = initial_field.reshape(1, *self.lattice_size)
        y_new = actual_evolution.reshape(1, *self.lattice_size)
        self.model.fit(X_new, y_new, epochs=training_steps, verbose=0)
        
        return predicted_evolution

# Example usage
lattice_size = (32, 32)
field = ScalarField(lattice_size, lambda_interaction=0.1, boundary='periodic')

X_train, y_train = generate_dataset(field, samples=5000, evolution_steps=20, noise_std=0.05)

model = build_model(lattice_size)
model.summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop, lr_schedule])

# Use the FieldEvolutionPredictor class
predictor = FieldEvolutionPredictor(model, lattice_size, field)
new_field = np.random.rand(*lattice_size)
predicted_evolution = predictor.predict_evolution(new_field)
field.visualize(predicted_evolution, "Predicted Evolution using Neural Network")

# Compare with the actual evolution without re-training
field.field = new_field
actual_evolution = field.evolve(steps=20)
field.visualize(actual_evolution, "Actual Evolution")
