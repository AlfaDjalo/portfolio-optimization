import tensorflow as tf

class BaseKerasModel(tf.keras.Model):
    """ 
    Base class for all keras models.
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.history = None

    def build_model(self, window_generator):
        """ Build the specific model architecture using window_generator to determine shapes. """
        raise NotImplementedError("Subclasses must implement build_model")

    def compile_model(self, optimizer='adam', loss='mse', metrics=['mae']):
        """ Compile the model with the specified parameters. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, window_generator, epochs=20, patience=25, verbose=1):
        """ Train the model using WindowGenerator. """
        if self.model is None:
            # Auto-build the model from window_generator
            self.build_model(window_generator)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                )

        self.history = self.model.fit(
            window_generator.train,
            epochs=epochs,
            validation_data=window_generator.val,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        return self.history

    def predict(self, inputs):
        """ Make predictions. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        return self.model(inputs).numpy()

    def __call__(self, inputs):
        """ Make the model callable. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        return self.model(inputs)

    def evaluate(self, window_generator):
        """ Evaluate on test data. """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        # return {'loss': 0.0, 'mae': 0.0}
        return self.model.evaluate(window_generator.test, verbose=0)


class KerasMLP(BaseKerasModel):
    """Multi-layer perceptron (feedforward neural network)."""
    
    def __init__(self, hidden_units=[64, 32], **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
    
    def build_model(self, window_generator):
        input_shape = (window_generator.input_width, len(window_generator.column_indices))

        num_labels = len(window_generator.label_columns) if window_generator.label_columns else 1
        output_size = window_generator.label_width * num_labels

        layers = [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten()
        ]

        for units in self.hidden_units:
            layers.extend([
                tf.keras.layers.Dense(units, activation='relu'),
                tf.keras.layers.Dropout(0.2)
            ])
        
        layers.append(tf.keras.layers.Dense(output_size, name='output'))
        self.model = tf.keras.Sequential(layers)





class KerasSkewedTDistribution(BaseKerasModel):
    """Custom MLP for generating skewed-t distribution parameters."""

    def __init__(self, ):
        super().__init__(**kwargs)
        return


    def build_model(self, ):
        return

    def compile_model(self, ):
        return
        