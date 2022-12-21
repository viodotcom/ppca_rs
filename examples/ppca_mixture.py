import numpy as np

from ppca_rs import PPCAMix, PPCAMixTrainer, Dataset
from ppca_rs.ppca_rs import PPCAModel

real_model = PPCAMix(
    [
        PPCAModel(
            transform=np.matrix([[1, 0, 0], [0, 0, 1]], dtype="float64").T,
            isotropic_noise=0.1,
            mean=np.array([[1, 1, 1]], dtype="float64").T,
        ),
        PPCAModel(
            transform=np.matrix([[1, 1, 0], [1, 0, 1]], dtype="float64").T,
            isotropic_noise=0.1,
            mean=np.array([[0, 1, 0]], dtype="float64").T,
        ),
    ],
    log_weights=np.log([0.33333, 0.66667]),
)

sample = real_model.sample(100, 0.1)

PPCAMixTrainer(sample).train(n_models=1, state_size=2, n_iters=30)
print()
PPCAMixTrainer(sample).train(n_models=2, state_size=2, n_iters=30)
print()
PPCAMixTrainer(sample).train(n_models=3, state_size=2, n_iters=30)
print()
model = PPCAMixTrainer(sample).train(n_models=4, state_size=2, n_iters=30)

model.smooth(sample)
model.extrapolate(sample)
model.infer(sample)
