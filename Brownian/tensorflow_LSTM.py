import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a simple time series dataset
dfLoad = pd.read_csv('..//data//GOLDBEES_SMALL.csv')
data = np.sin(np.linspace(0, 100, 1000))  # Sine wave data
dfLoad['time'] = pd.to_datetime(dfLoad['time'])
dfLoad.sort_values(by='time', inplace=True)

df = dfLoad[['intc']]
# df = dfLoad['intc']
print(df)
df.rename(columns={'intc': 'value'}, inplace=True)
df.reset_index(drop=True, inplace=True)
# df.columns = ['intc', 'value']
plt.plot(df['value'])
plt.show()




#####TENSOR START###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define the model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
