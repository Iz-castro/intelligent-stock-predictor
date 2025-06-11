from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=1e-6)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
