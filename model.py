import tensorflow as tf

# Load the model from .h5 file
model = tf.keras.models.load_model('best_plant_disease_model.h5')

# Print model summary to verify
model.summary()
