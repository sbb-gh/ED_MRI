import simulations
import numpy as np
from dmipy.data import saved_acquisition_schemes
import timeit


base_save = "" # TODO set save directory for data

np.random.seed(0)
scheme = saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
nsamples_train = 10**5
nsamples_val = nsamples_train // 10
nsamples_test = nsamples_train // 10

start_timer = timeit.default_timer()
train_sims = simulations.noddi(nsamples_train, scheme)
val_sims = simulations.noddi(nsamples_val, scheme)
test_sims = simulations.noddi(nsamples_test, scheme)
print("simulations time", timeit.default_timer() - start_timer)

data = dict(
    train=train_sims[0],
    train_tar=train_sims[1],
    val=val_sims[0],
    val_tar=val_sims[1],
    test=test_sims[0],
    test_tar=test_sims[1],
)

nsamples_train_K = nsamples_train // 1000
save_name = base_save + "noddi_simulations"
input(f"About to save in {save_name} press enter")
np.save(save_name, data)

print("EOF", __file__)
