import tensorflow as tf
from transformers import Transformer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from dataloader import TTdata


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def read_transform(csv_path):
    df = pd.read_csv(csv_path)
    df.drop(df.columns[0], axis=1, inplace=True)
    df[:] = scaler.fit_transform(df)
    return df


scaler = MinMaxScaler()

csv_path = 'Data/energydata_complete.csv'
train_test_ratio = 0.8

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
input_length = 20
output_length = 3
batch_size = 16

data = read_transform(csv_path)
split = int(data.shape[0] * train_test_ratio)

X = data.copy()
Y = data.drop(['Appliances', 'lights', 'Windspeed', 'Visibility'], axis=1)

input_size = X.shape[1]
num_predicted_features = Y.shape[1]

X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

train_batches = TTdata(X_train, Y_train, batch_size, input_length, output_length)
val_batches = TTdata(X_val, Y_val, batch_size, input_length, output_length)

# learning_rate = CustomSchedule(d_model)
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
out_features = Y_val.shape[1]

model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    out_features=out_features,
    dropout_rate=dropout_rate)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(train_batches,epochs=100,validation_data=val_batches)
