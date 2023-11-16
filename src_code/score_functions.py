import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import BallTree

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def variance_score(zs_posterior):
	# zs_posterior: (V, B, dz)
	return torch.var(zs_posterior, dim=0).sum(dim=-1)

def variance_all_timesteps_score(zs_posterior):
	# zs_posterior: (V, T, B, dz)
	return torch.var(zs_posterior, dim=0).sum(dim=[0, 2])

def logLikelihood(Xv, mu_x_v, var_x_v):
	# X: (batch_sz, dv)
	# mu_x: (batch_sz, dv)
	# var_x: (batch_sz, dv)
	# No worry, it broadcasts var_x on X and mu_x
	B, dv = Xv.size()
	pi = torch.tensor(np.pi)
	const = dv*torch.log(2*pi)
	llh = torch.log(var_x_v) + ((Xv-mu_x_v)**2)/var_x_v
	return -0.5 * (const + torch.sum(llh, dim=1))

def logLikelihood_all_timesteps(Xv, mu_x_v, var_x_v):
	# X: (T, B, dv)
	# mu_x: (T, B, dv)
	# var_x: (T, B, dv)
	# No worry, it broadcasts var_x on X and mu_x
	T, B, dv = Xv.size()
	pi = torch.tensor(np.pi)
	const = dv*torch.log(2*pi)
	llh = torch.log(var_x_v) + ((Xv-mu_x_v)**2)/var_x_v
	return -0.5 * (const + torch.sum(llh, dim=[0, 2]))

def logllh_crossview_score(zs, X, decoders):
	# zs: VxNxdz
    V, _, _ = zs.size()
    scores = 0
    for i in range(V):
        for j in range(V):
            if i != j:
                mu_x_v = decoders[j].mu_emission(zs[i])
                var_x_v = decoders[j].sigma_emission #(zs[i])
                # From the log_likelihood is modified at the beginning of this file
                scores -= logLikelihood(X[j], mu_x_v, var_x_v)
    return scores

def logllh_crossview_all_timesteps_score(zs, X, decoders):
	# zs: (V, T, N, dz)
    V, _, _, _ = zs.size()
    scores = 0
    for i in range(V):
        for j in range(V):
            if i != j:
                mu_x_v = decoders[j].mu_emission(zs[i])
                var_x_v = decoders[j].sigma_emission(zs[i])
                # From the log_likelihood is modified at the beginning of this file
                scores -= logLikelihood_all_timesteps(X[j], mu_x_v, var_x_v)
    return scores

def logllh_oneVAE_crossview_score(zs, X, mu_emissions, sigma_emissions):
	# zs: VxNxdz
    V, _, _ = zs.size()
    scores = 0
    for i in range(V):
        for j in range(V):
            if i != j:
                mu_x_v = mu_emissions[j](zs[i])
                var_x_v = sigma_emissions[j](zs[i])
                # From the log_likelihood is modified at the beginning of this file
                scores -= logLikelihood(X[j], mu_x_v, var_x_v)
    return scores

def logllh_oneVAE_crossview_all_timesteps_score(zs, X, mu_emissions, sigma_emissions):
	# zs: (V,T,N,dz)
    V, _, _, _ = zs.size()
    scores = 0
    for i in range(V):
        for j in range(V):
            if i != j:
                mu_x_v = mu_emissions[j](zs[i])
                var_x_v = sigma_emissions[j](zs[i])
                # From the log_likelihood is modified at the beginning of this file
                scores -= logLikelihood_all_timesteps(X[j], mu_x_v, var_x_v)
    return scores

def logllh_oneVAE_score(zs, X, mu_emissions, sigma_emissions):
	# zs: VxNxdz
    V, _, _ = zs.size()
    scores = 0
    for i in range(V):
        mu_x_v = mu_emissions[i](zs[i])
        var_x_v = sigma_emissions[i](zs[i])
        # From the log_likelihood is modified at the beginning of this file
        scores -= logLikelihood(X[i], mu_x_v, var_x_v)
    return scores

def logllh_oneVAE_all_timesteps_score(zs, X, mu_emissions, sigma_emissions):
	# zs: (V, T, N, dz)
    V, _, _, _ = zs.size()
    scores = 0
    for i in range(V):
        mu_x_v = mu_emissions[i](zs[i])
        var_x_v = sigma_emissions[i](zs[i])
        # From the log_likelihood is modified at the beginning of this file
        scores -= logLikelihood_all_timesteps(X[i], mu_x_v, var_x_v)
    return scores

def logllh_pxz_crossview_score(zs, X, mu_emissions, sigma_emissions, gated_transistion, K):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    # K = 5
    scores = 0
    for i in range(V):
        for j in range(V):
            if i != j:
                scores -= logLikelihood(X[j], mu_emissions[j](zs[i, -1]), sigma_emissions[j](zs[i, -1]))
                for k in range(2, K+1):
                    z_t_k = gated_transistion(zs[i, -k])['mu']
                    scores -= logLikelihood(X[j], mu_emissions[j](z_t_k), sigma_emissions[j](z_t_k))
    return scores


def logllh_pxz_AB_score(zs, X, mu_emissions, sigma_emissions, gated_transistion, K):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    # K = 5
    scores = 0
    for i in range(V):
        A = 0
        for j in range(V):
            if i != j:
                A -= logLikelihood(X[j], mu_emissions[j](zs[i, -1]), sigma_emissions[j](zs[i, -1]))
                for k in range(2, K+1):
                    z_t_k = gated_transistion(zs[i, -k])['mu']
                    A -= logLikelihood(X[j], mu_emissions[j](z_t_k), sigma_emissions[j](z_t_k))
        B = -logLikelihood(X[i], mu_emissions[i](zs[i, -1]), sigma_emissions[i](zs[i, -1]))
        for k in range(2, K+1):
            z_t_k = gated_transistion(zs[i, -k])['mu']
            B -= logLikelihood(X[i], mu_emissions[i](z_t_k), sigma_emissions[i](z_t_k))
        scores += A + B
    return scores


def logllh_pxz_AB_pair_score(zs, X, mu_emissions, sigma_emissions, gated_transistion, K):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    # K = 5
    scores = 0
    for i in range(V):
        scores_i = 0
        for j in range(V):
            if i != j:
                A = logLikelihood(X[j], mu_emissions[j](zs[i, -1]), sigma_emissions[j](zs[i, -1]))
                for k in range(2, K+1):
                    z_t_k = gated_transistion(zs[i, -k])['mu']
                    A += logLikelihood(X[j], mu_emissions[j](z_t_k), sigma_emissions[j](z_t_k))
                B = logLikelihood(X[j], mu_emissions[j](zs[j, -1]), sigma_emissions[j](zs[j, -1]))
                for k in range(2, K+1):
                    z_t_k = gated_transistion(zs[j, -k])['mu']
                    B += logLikelihood(X[j], mu_emissions[j](z_t_k), sigma_emissions[j](z_t_k))
                scores_i += torch.abs(A-B)
        scores += scores_i
    return scores


def logllh_pxz_crossview_minMaxScaled_score(zs, X, mu_emissions, sigma_emissions, gated_transistion):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    scores = []
    for i in range(V):
        score_v = 0
        for j in range(V):
            if i != j:
                mu_x_v = mu_emissions[j](zs[i, -1])
                var_x_v = sigma_emissions[j](zs[i, -1])
                z_prev = zs[i, -2]
                latent_prior = gated_transistion(z_prev)
                mu_z_prior = latent_prior['mu']
                std_z_prior = latent_prior['std']
                eps = torch.randn_like(mu_z_prior, device=device)
                z_prior = mu_z_prior# + std_z_prior * eps
                score_v -= logLikelihood(X[j], mu_x_v, var_x_v)    # p(x_t|z_t)
                score_v -= logLikelihood(X[j], mu_emissions[j](z_prior), sigma_emissions[j](z_prior))    # p(x_t|z_t-1)
                z_t_2 = gated_transistion(zs[i, -3])['mu']
                score_v -= logLikelihood(X[j], mu_emissions[j](z_t_2), sigma_emissions[j](z_t_2))    # p(x_t|z_t-2)
                z_t_3 = gated_transistion(zs[i, -4])['mu']
                score_v -= logLikelihood(X[j], mu_emissions[j](z_t_3), sigma_emissions[j](z_t_3))    # p(x_t|z_t-3)
                z_t_4 = gated_transistion(zs[i, -5])['mu']
                score_v -= logLikelihood(X[j], mu_emissions[j](z_t_4), sigma_emissions[j](z_t_4))    # p(x_t|z_t-4)
                
        scaler = MinMaxScaler()
        scores.append(torch.tensor(scaler.fit_transform(score_v.cpu().numpy().reshape(-1,1)), device=device))
    return torch.stack(scores, dim=0).mean(dim=0).view((-1,))


def logllh_pxz_score(zs, X, mu_emissions, sigma_emissions, gated_transistion, K):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    scores = 0
    for v in range(V):
        for k in range(1, K+1):
            scores -= logLikelihood(X[v], mu_emissions[v](zs[v, -k]), sigma_emissions[v](zs[v, -k]))
    return scores

def logllh_pxz_pairwise_score(zs, X, mu_emissions, sigma_emissions, gated_transistion, K):
    # zs: (V, T, B, dz)
    V, T, B, dz = zs.size()
    scores = 0
    
    
    sum = 0;
    for v1 in range(V):
        sum += zs[v1, -1]
    for v1 in range(V):
        scores -= logLikelihood(
                    X[v1],
                    mu_emissions[v1](sum/V),
                    sigma_emissions[v1](sum/V)
                )
    '''
    for v1 in range(V):
        maxs = torch.ones(B)
        for v2 in range(V):    
            for k in range(1, K+1):
                #scores -= logLikelihood(
                #    X[v1],
                #    mu_emissions[v1](zs[v2, -k]),
                #    sigma_emissions[v1](zs[v2, -k])
                #)
                llh = logLikelihood(
                    X[v1],
                    mu_emissions[v1](zs[v2, -k]),
                    sigma_emissions[v1](zs[v2, -k])
                )
                
                for i in range(llh.shape[0]):
                    if(maxs[i]>llh[i].item()): maxs[i] = llh[i].item()
        scores -= maxs
    '''
    
    return scores

def logllh_pxz_all_timesteps_score(zs, X, mu_emissions, sigma_emissions, gated_transistion):
    # zs: (V, B, dz)
    V, B, dz = zs.size()
    scores = 0
    for i in range(V):
        mu_x_v = mu_emissions[i](zs[i])
        var_x_v = sigma_emissions[i](zs[i])
        mu_z_prior, std_z_prior = [], []
        for t in range(T):
            z_prev = torch.zeros(B, dz, device=device) if t == 0 else zs[i, t-1]
            out_prior = gated_transistion(z_prev)
            mu_z_prior.append(out_prior['mu'])
            std_z_prior.append(out_prior['std'])
        mu_z_prior = torch.stack(mu_z_prior)
        std_z_prior = torch.stack(std_z_prior)
        scores -= logLikelihood(X[i], mu_x_v, var_x_v)    # sum_t(p(x_t|z_t))
        scores -= logLikelihood(X[i], mu_z_prior, std_z_prior)   # sum_t(p(z_t|z_t-1))
    return scores

def euclidean_dist(a, b):
    return ((a-b)**2).sum().sqrt()

def find_neighbors(z, radius, zs):
    # z: dz
    # radius: scalar
    # zs: V x N x dz
    V, N, dz = zs.size()
    neighbors = []
    for v in range(V):
        for n in range(N):
            if euclidean_dist(z, zs[v,n]) < radius:
                neighbors.append(zs[v,n])
    return neighbors

def calc_radius(zs):
    V, N, dz = zs.size()
    radius = 0
    count = 0
    for v1 in range(V):
        for n1 in range(N):
            for v2 in range(v1, V):
                for n2 in range(n1, N):
                    if v1 != v2 or n1 != n2:
                        radius += euclidean_dist(zs[v1, n1], zs[v2, n2])
                        count += 1
    return radius/count

def in_set(element, set_):
    flag = False
    for e in set_:
        if (element == e).all().item():
            flag = True
            break
    return flag

def jaccard_of_sample(zs_n, zs, radius):
    # zs_n: views of sample n_th (V, dz)
    # zs: (V, N, dz)
    V, N, dz = zs.size()
    score = 0
    for v1 in range(V):
        for v2 in range(v1+1, V):
            neighbors1 = find_neighbors(zs_n[v1], radius, zs)
            neighbors2 = find_neighbors(zs_n[v2], radius, zs)
            intersection_count = len([z for z in neighbors1 if in_set(z, neighbors2)])
            union_count = len(neighbors1) + len(neighbors2) - intersection_count
            score += intersection_count/union_count
    return score

def jaccard_score(zs):
    # zs: (V, N, dz)
    V, N, dz = zs.size()
    radius = calc_radius(zs)
    jac_scores = [torch.tensor(jaccard_of_sample(zs[:,n], zs, radius)) for n in range(N)]
    return torch.stack(jac_scores).to(device)

def jaccard_of_sample_all_timesteps(zs_n, zs, radius):
    # zs_n: views of sample n_th (V, T, dz)
    # zs: (V, T, N, dz)
    V, T, N, dz = zs.size()
    score = 0
    for v1 in range(V):
        for v2 in range(v1+1, V):
            for t in range(T):
                neighbors1 = find_neighbors(zs_n[v1, t], radius, zs[:, t])
                neighbors2 = find_neighbors(zs_n[v2, t], radius, zs[:, t])
                intersection_count = len([z for z in neighbors1 if in_set(z, neighbors2)])
                union_count = len(neighbors1) + len(neighbors2) - intersection_count
                score += intersection_count/union_count
    return score

def jaccard_all_timesteps_score(zs):
    # zs: (V, T, N, dz)
    V, T, N, dz = zs.size()
    radius = calc_radius(zs[:,-1])
    jac_scores = [torch.tensor(jaccard_of_sample_all_timesteps(zs[:,:,n], zs, radius)) for n in range(N)]
    return torch.stack(jac_scores).to(device)

def pij_balltree(X, k):
    # X: (N, dx)
    # return (N, N-1)
    N = X.size(0)
    # dist_ts = torch.cdist(X, X, p=2.0)
    # print(dist_ts.size())
    X_np = X.numpy()
    print("Calculating ball tree")
    tree = BallTree(X_np, leaf_size=int(N*2/3), metric='euclidean') # leaf_size <= N <= 2*leaf_size
    
    dist, ind = tree.query(X, k=k+1)
    # print(dist.shape)
    # Remove the first column since it is the considered point itself
    dist_ts = torch.tensor(dist[:, 1:], dtype=torch.float32) + 1e-24  # (N, k)
    print(dist_ts.size())
    ind_np = ind[:, 1:]                          # (N, k)
    
    P_ts = torch.exp(-dist_ts) / torch.sum(torch.exp(-dist_ts), dim=1, keepdim=True)
    
    return P_ts, ind_np
    # pX = []
    # for i in range(N):
    #     row = torch.zeros((N,), device=torch.device("cpu")) + 1e-24
    #     topK_dist_indices = torch.sort(dist_ts[i], descending=False)[1][
    #         1 : k + 1
    #     ]  # get index
    #     for j in topK_dist_indices:
    #         # for j in range(N):
    #         # denominator = torch.sum(torch.stack([torch.exp(-dist_ts[i, k]) for k in range(N) if k != i]))
    #         denominator = torch.sum(
    #             torch.exp(-dist_ts[i, topK_dist_indices]) + 1e-24
    #         )  # - torch.exp(-dist_ts[i, i])
    #         row[j] = torch.exp(-dist_ts[i, j]) / denominator
            
    #     pX.append(row)
    # pX_ts = torch.stack(pX, dim=0)
    # print(pX_ts)
    # return pX_ts + 1e-24

# def pij_fast(X, k):
    


def KL_PQ(pX, qZ):
    p_X, ind_X = pX
    q_Z, ind_Z = qZ
    N = p_X.size(0)
    scores = []
    for i in range(N):
        # if i % 1000 == 0:
        #     print(i)
        tmp_X = torch.zeros((N,), device=torch.device('cpu')) + 1e-24
        tmp_X[ind_X[i]] = p_X[i]
        tmp_Z = torch.zeros((N,), device=torch.device('cpu')) + 1e-24
        tmp_Z[ind_Z[i]] = q_Z[i]
        scores.append(torch.sum(tmp_X * torch.log(tmp_X / tmp_Z)))
    
    # score_ = torch.sum(pX*torch.log(pX / qZ), dim=1)
    
    # for i in range(N):
    #     # score = 0
    #     # for j in range(N):
    #     #     if j != i:
    #     #         score += pX[i, j] * torch.log(pX[i, j] / qZ[i, j])
    #     # scores.append(score)
    #     score_[i] -= pX[i,i] * torch.log(pX[i,i] / qZ[i,i])
    #     # if i%100 == 0:
    #     #     # print(score)
    #     #     print(score_[i])
        
    # return score_
    return torch.stack(scores, dim=0)


def sne_score_fn(Z, X, k=30):
    # X: (V, N, dv)
    # Z: (V, N, dz)
    Z = Z.cpu()
    X = [Xv.cpu() for Xv in X]
    
    pZ = [pij_balltree(Zv, k) for Zv in Z]
    pX = [pij_balltree(Xv, k) for Xv in X]
    # scores_1kl = 0
    scores_crossview_1kl = 0
    # for v, Xv in enumerate(X):
    #     pXv = pij_fast(Xv, k)
    #     # pXv = pij(Xv, k)
    #     scores_1kl += KL_PQ(pZ, pXv)
    #     # scores_2kl += scores_1kl + KL_PQ(pXv, pZ)
    score_view_stack = []
    for v1 in range(len(X)):
        print(f"view {v1+1}")
        score_views = 0
        for v2 in range(len(X)):
            # if v2 != v1:
            #     scores_crossview_1kl += KL_PQ(pZ[v1], pX[v2])
            # else:
            #     scores_1kl += KL_PQ(pZ[v1], pX[v2])
            # scores_crossview_1kl += KL_PQ(pZ[v1], pX[v2])
            score_views += KL_PQ(pZ[v1], pX[v2])
        score_view_stack.append(score_views)
    score_view_stack = torch.stack(score_view_stack, dim=1)
    scores_crossview_1kl = torch.max(score_view_stack, dim=1)[0]    # only get max values
    # mask = torch.isfinite(scores_1kl)
    # print(scores_1kl)
    # print(pZ[mask==False])
    # print(pXv[mask==False])
    return {
        # 'sne_1kl': scores_1kl,
        'sne_crossview_1kl': scores_crossview_1kl
    }