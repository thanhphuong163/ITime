import torch
import numpy as np

def KLD_w_log(mu1, logvar1, mu2, logvar2):
    # (timesteps, batch_sz, latent_sz)
    T, B, dz = mu1.size()
    kld = logvar2-logvar1 + torch.exp(logvar1-logvar2) + torch.exp(torch.log((mu2-mu1)**2 + 1e-20)-logvar2)
    kld = torch.sum(kld, dim=(0, 2)) - T*dz
    return 0.5 * kld

def KLD(mu1, var1, mu2, var2):
    # (timesteps, batch_sz, latent_sz)
    T, B, dz = mu1.size()
    kld = torch.log(var2/(var1)) + var1/(var2) + ((mu2-mu1)**2)/(var2)
    kld = torch.sum(kld, dim=(0, 2)) - T*dz
    return 0.5 * kld

def log_likelihood_w_log(X, mu_x, logvar_x):
    # X: (timesteps, batch_sz, dv)
    # mu_x: (timesteps, batch_sz, dv)
    # logvar_x: (timesteps, batch_sz, dv)
    # No worry, it broadcasts var_x on X and mu_x
    T, B, dv = X.size()
    pi = torch.tensor(np.pi)
    const = T*dv*torch.log(2*pi)
    #llh = torch.log(var_x) + ((X-mu_x)**2)/var_x
    llh = logvar_x + torch.exp(torch.log((X-mu_x)**2 + 1e-20)-logvar_x)
    return -0.5 * (const + torch.sum(llh, dim=(0, 2)))

def log_likelihood(X, mu_x, var_x):
    # X: (timesteps, batch_sz, dv)
    # mu_x: (timesteps, batch_sz, dv)
    # var_x: (timesteps, batch_sz, dv)
    # No worry, it broadcasts var_x on X and mu_x
    T, B, dv = X.size()
    pi = torch.tensor(np.pi)
    const = T*dv*torch.log(2*pi)
    llh = torch.log(var_x) + ((X-mu_x)**2)/var_x
    return -0.5 * (const + torch.sum(llh, dim=(0, 2)))

def negELBO_w_log(X, mu_x, logvar_x, mu1, logvar1, mu2, logvar2):
    kld = KLD_w_log(mu1, logvar1, mu2, logvar2)
    llh = log_likelihood_w_log(X, mu_x, logvar_x)
    neg_elbo = kld - llh
    return {
        'negELBO': neg_elbo,
        'kld': kld,
        'logllh': llh
    }

def negELBO(X, mu_x, var_x, mu1, var1, mu2, var2):
    kld = KLD(mu1, var1, mu2, var2)
    llh = log_likelihood(X, mu_x, var_x)
    neg_elbo = kld - llh
    return {
        'negELBO': neg_elbo,
        'kld': kld,
        'logllh': llh
    }