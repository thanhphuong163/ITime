import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import auroc
from loss_functions import negELBO, negELBO_w_log
from score_functions import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class STLROneVAE(nn.Module):
    def __init__(self, view_szes, latent_sz, rnn_hidden_sz, emission_hidden_sz, transition_hidden_sz,
                 n_layers=1, dropout=0., nonlinearity='tanh'):
        super(STLROneVAE, self).__init__()

        self.view_szes = view_szes
        self.latent_sz = latent_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.emission_hidden_sz = emission_hidden_sz
        self.transition_hidden_sz = transition_hidden_sz
        self.in_sz = max(view_szes)
        self.total_sz = sum(view_szes)
        
        # Define model
        self.filters1 = nn.ModuleList([
            nn.Sequential(
                #nn.Linear(view_sz, 200),
                #nn.ReLU(),
                #nn.Linear(200, self.in_sz)
                #tuan
                nn.Linear(view_sz, self.in_sz)
                #endtuan
            )
            for view_sz in self.view_szes
        ])

        self.filters2 = nn.ModuleList([
            nn.Sequential(
                #nn.Linear(self.total_sz - view_sz, 200),
                #nn.ReLU(),
                #nn.Linear(200, self.in_sz)
                #tuan
                nn.Linear(self.total_sz - view_sz, self.in_sz)
                #endtuan
            )
            for view_sz in self.view_szes
        ])

        self.encoder = BRNN(
                in_sz=self.in_sz*2,
                # in_sz=self.in_sz,
                latent_sz=self.latent_sz,
                rnn_hidden_sz=self.rnn_hidden_sz,
                n_layers=n_layers,
                nonlinearity='tanh',
                bias=True,
                batch_first=False,
                dropout=dropout,
                bidirectional=True
                )

        # self.encoders = nn.ModuleList([
        #     BRNN(
        #         in_sz=view_sz,
        #         latent_sz=self.latent_sz,
        #         rnn_hidden_sz=self.rnn_hidden_sz,
        #         n_layers=n_layers,
        #         nonlinearity='tanh',
        #         bias=True,
        #         batch_first=False,
        #         dropout=dropout,
        #         bidirectional=True
        #     )
        #     for view_sz in self.view_szes
        # ])

        self.decoder = DMM(
                latent_sz=self.latent_sz,
                emission_hidden_sz=self.emission_hidden_sz,
                transition_hidden_sz=self.transition_hidden_sz,
                view_sz=self.latent_sz
            )

        self.mu_emissions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_sz, emission_hidden_sz),
                nn.ReLU(),
                #tuan
                #nn.Linear(emission_hidden_sz, emission_hidden_sz),
                #nn.ReLU(),
                #endtuan
                nn.Linear(emission_hidden_sz, view_sz),
            )
            for view_sz in self.view_szes
        ])

        self.sigma_emissions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_sz, emission_hidden_sz),
                nn.ReLU(),
                #tuan
                #nn.Linear(emission_hidden_sz, emission_hidden_sz),
                #nn.ReLU(),
                #endtuan
                nn.Linear(emission_hidden_sz, view_sz),
                nn.Softplus()
            )
            for view_sz in self.view_szes
        ])

        # self.sigma_emissions = nn.ParameterList([
        # 	nn.Parameter(torch.ones(1,1,view_sz)*0.01, requires_grad=False)
        # 	for view_sz in self.view_szes
        # ])
    
    def get_loss(self, Xv, mu_x_v, std_x_v, mu_post, std_post, mu_prior, std_prior):
        loss = negELBO(
            Xv, mu_x_v, std_x_v**2,
            mu_post, std_post**2,
            mu_prior, std_prior**2
        )
        return loss

    def get_score(self, zs_post, X):
        # X: V x (T,N,dv)
        # zs_post: (V,T,N,dz)
        scores = {
            'logllh_score': logllh_pxz_score(zs_post, [Xv[-1] for Xv in X], self.mu_emissions, self.sigma_emissions, self.decoder.gated_transition, 2),
            }
        return scores

    def forward(self, X):
        # X: V x (B, T, dv)
        output = {
            'z_post': [],
            'mu_post': [],
            'std_post': [],
            'mu_prior': [],
            'std_prior': [],
            'mu_x': [],
            'std_x': [],
            }
        loss = {'negELBO': None, 'kld': None, 'logllh': None}
        # Calculation
        # for Xv, encoder, decoder in zip(X, self.encoders, self.decoders):
        for v, (Xv, filter1, filter2, mu_emission_, sigma_emission_) in enumerate(zip(X, self.filters1, self.filters2, self.mu_emissions, self.sigma_emissions)):
            X_v = filter1(Xv.transpose(0, 1))
            X_rest = filter2(torch.concat([X[i].transpose(0, 1) for i in range(len(X)) if i != v], dim=-1))
            # input_X = X_v + X_rest
            input_X = torch.concat([X_v, X_rest], dim=-1)
            # input_X = X_v
            enc_out = self.encoder(input_X)
            dec_out = self.decoder(enc_out['z_post'])
            dec_out['mu_x'] = mu_emission_(enc_out['z_post'])
            dec_out['std_x'] = sigma_emission_(enc_out['z_post'])
            loss_v = self.get_loss(
                Xv.transpose(0, 1), dec_out['mu_x'], dec_out['std_x'],
                enc_out['mu_post'], enc_out['std_post'],
                dec_out['mu_prior'], dec_out['std_prior'])
            for key in enc_out.keys():
                output[key].append(enc_out[key])
            for key in dec_out.keys():
                output[key].append(dec_out[key])
            for key in loss_v.keys():
                if loss[key] is None:
                    loss[key] = loss_v[key]
                else:
                    loss[key] += loss_v[key]
        return output, loss

class GaussianNN(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(GaussianNN, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.mu_net = nn.Sequential(
            nn.Linear(in_sz, out_sz, bias=True)
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(in_sz, out_sz, bias=True),
            nn.Softplus()
        )
    
    @staticmethod
    def reparameterization(mu, std):
        eps = torch.randn_like(mu, device=device)
        z = mu + std * eps
        return z

    def forward(self, X):
        # X: (B, dx)
        mu = self.mu_net(X)
        std = self.sigma_net(X)
        z = self.reparameterization(mu, std)
        return {'mu': mu,'std': std, 'z': z}

class BRNN(nn.Module):
    def __init__(
        self,
        in_sz,
        latent_sz,
        rnn_hidden_sz,
        n_layers=1,
        nonlinearity='tanh',
        bias=True,
        batch_first=False,
        dropout=0.,
        bidirectional=True
        ):
        super(BRNN, self).__init__()
        self.in_sz = in_sz
        self.latent_sz = latent_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.n_layers = n_layers
        self.D = 2 if bidirectional else 1
        
        # Define model

        self.rnn = nn.RNN(
            input_size=in_sz,
            hidden_size=rnn_hidden_sz,
            num_layers=n_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)
        '''
        self.rnn = nn.LSTM(
            input_size=in_sz,
            hidden_size=rnn_hidden_sz,
            num_layers=n_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional)
        '''
        self.gaussian_combine = GaussianNN(rnn_hidden_sz, latent_sz)
        self.transition_net = nn.Sequential(
            nn.Linear(latent_sz, rnn_hidden_sz),
                nn.Tanh()
            # nn.ReLU()
        )
    
    def combine_h(self, h):
        # h: (T, B, 2*dh)
        T, B, _ = h.size()
        h_left = h[:,:,:self.rnn_hidden_sz]
        h_right = h[:,:,self.rnn_hidden_sz:]
        z_prev = torch.zeros(B, self.latent_sz, device=device)
        z, mu, std, h_combined = [], [], [], []
        for t in range(T):
            tmp = self.transition_net(z_prev)
            h_t = (1/3.0) * (tmp + h_left[t] + h_right[t])
            # h_t = torch.concat([tmp, h_left[t], h_right[t]], dim=-1)
            out = self.gaussian_combine(h_t)
            z_prev = out['z']
            h_combined.append(h_t)
            mu.append(out['mu'])
            std.append(out['std'])
            z.append(out['z'])
        return {
            'h_combined': torch.stack(h_combined),
            'mu': torch.stack(mu),
            'std': torch.stack(std),
            'z': torch.stack(z)
        }
    
    def forward(self, X):
        # X: (T, B, dx)
        T, B, dx = X.size()
        h_0 = torch.zeros(self.D*self.n_layers, B, self.rnn_hidden_sz, device=device)
        h, h_n = self.rnn(X, h_0)
        #c_0 = torch.zeros(self.D*self.n_layers, B, self.rnn_hidden_sz, device=device)
        #h, (h_n, c_n) = self.rnn(X, (h_0, c_0))
        out = self.combine_h(h)
        return {
            'z_post': out['z'],
            'mu_post': out['mu'],
            'std_post': out['std']
            }

class DMM(nn.Module):
    def __init__(self, latent_sz, emission_hidden_sz, transition_hidden_sz, view_sz):
        super(DMM, self).__init__()
        self.latent_sz = latent_sz # latent_sz
        self.emission_hidden_sz = emission_hidden_sz
        self.transition_hidden_sz = transition_hidden_sz
        self.view_sz = view_sz # x size
        # Define model
        self.gating_unit = nn.Sequential(
            nn.Linear(latent_sz, transition_hidden_sz),
            nn.ReLU(),
            nn.Linear(transition_hidden_sz, latent_sz),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(latent_sz, transition_hidden_sz),
            nn.ReLU(),
            nn.Linear(transition_hidden_sz, latent_sz),
            nn.Identity()
        )
        self.mu_net = nn.Sequential(
            nn.Linear(latent_sz, latent_sz)
        )
        self.sigma_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_sz, latent_sz),
            nn.Softplus()
        )
        self.z_init = nn.Parameter(torch.zeros(1, self.latent_sz, device=device), requires_grad=True)
    
    def mu_transition(self, z_prev, h_t, g_t):
        mu_t = (1 - g_t) * self.mu_net(z_prev) + g_t * h_t
        return mu_t
    
    def sigma_transition(self, h_t):
        std_t = self.sigma_net(h_t)
        return std_t
    
    def gated_transition(self, z_prev):
        g_t = self.gating_unit(z_prev)
        h_t = self.proposed_mean(z_prev)
        mu_t = self.mu_transition(z_prev, h_t, g_t)
        std_t = self.sigma_transition(h_t)
        return {'mu': mu_t,'std': std_t}
    
    def forward(self, Z):
        # Z: (T, B, dz)
        T, B, dz = Z.size()
        mu_z, std_z = [], []
        mu_x, std_x = [], []
        for t in range(T):
            z_prev = self.z_init.repeat(B, 1) if t == 0 else Z[t-1]
            out = self.gated_transition(z_prev)
            mu_z.append(out['mu'])
            std_z.append(out['std'])
        
        return {
            'mu_prior': torch.stack(mu_z),
            'std_prior': torch.stack(std_z),
            }