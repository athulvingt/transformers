import tensorflow as tf
import math
import numpy as np


class TTdata(tf.keras.utils.Sequence):
    def __init__(self, X, Y,
                 batch_size,
                 enc_seq_len,
                 target_seq_len):
        self.X = X
        self.Y = Y
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len
        self.data_len = self.X.shape[0] - self.enc_seq_len - self.target_seq_len
        self.batch_size = batch_size


    def __getitem__(self, batch_idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """

        rows = np.arange(batch_idx * self.batch_size, min(self.batch_size * (batch_idx + 1), self.data_len))
        src = np.array([self.X[index:index + self.enc_seq_len] for index in rows])
        trg = np.array(
            [self.Y[index + self.enc_seq_len - 1:index + self.target_seq_len - 1 + self.enc_seq_len] for index in rows])
        trg_y = np.array(
            [self.Y[index + self.enc_seq_len: index + self.enc_seq_len + self.target_seq_len] for index in rows])

        return (src, trg), trg_y

    def __len__(self):
        return math.ceil(self.data_len / self.batch_size)


if __name__ == "__main__":
    x = np.random.rand(19, 20)
    y = np.random.rand(19, 16)
    d = TTdata(x, y, 2, 5, 3)
    for i in d:
        print(i)
