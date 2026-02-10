
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


torch.set_default_dtype(torch.double) 

def forecast(X, y):
    
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(60.0),
                        variance=torch.tensor(0.5))

    optim = Adam({"lr": 0.01}) 

    pyro.clear_param_store()
    
    # Append the forecast length onto the known days. (16-day forecasts)
    plus_arr = np.max(X) + np.array([16., 32., 48.])
    
    Xtest_use = np.append(X, plus_arr)

    # Convert numpy arrays into Torch tensors (for quicker calculations)
    X2 = torch.from_numpy(X)
    y2 = torch.from_numpy(y - np.mean(y))
    Xtest_use2 = torch.from_numpy(Xtest_use)

    # Activate the module using the data and the kernel. Specify the noise
    gpr = gp.models.GPRegression(X2, y2, k1, noise=torch.tensor(0.01))

    # Stochastic Variational Inference to find the optimal fit for the data
    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())

    # Specify how many steps of the SVI
    num_steps = 10
    losses = np.empty(num_steps)

    # Step through the SVI
    for k in range(num_steps):
        losses[k] = svi.step()

    # Retrieve mean predictions
    with torch.no_grad():
        if isinstance(gpr, gp.models.VariationalSparseGP):
            mean, cov = gpr(Xtest_use2, full_cov=True)
        else:
            mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 
    
    mean = mean.detach().numpy() + np.mean(y)

    return Xtest_use, mean
