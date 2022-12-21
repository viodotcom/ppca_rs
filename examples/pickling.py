import pickle
import numpy as np

from ppca_rs import PPCAModel

model = PPCAModel(
    transform=np.matrix([[1, 1, 0], [1, 0, 1]], dtype="float64").T,
    isotropic_noise=0.1,
    mean=np.array([[0, 1, 0]], dtype="float64"),
)

ser = pickle.dumps(model)
de = pickle.loads(ser)

print(model)
print(de)
