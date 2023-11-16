import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import auroc
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from modules import *
from loss_functions import negELBO, negELBO_w_log
from score_functions import *
from utils import *

class LitModule(pl.LightningModule):
    def __init__(self, view_szes, latent_sz, rnn_hidden_sz, emission_hidden_sz, transition_hidden_sz,
                 output_dir, n_layers=1, dropout=0., nonlinearity='tanh', lr=1e-3):
        super(LitModule, self).__init__()
        # Hyperparams
        self.view_szes = view_szes
        self.latent_sz = latent_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.emission_hidden_sz = emission_hidden_sz
        self.transition_hidden_sz = transition_hidden_sz
        self.n_layers=n_layers
        self.dropout=dropout
        self.nonlinearity=nonlinearity
        self.lr = lr
        self.output_dir = output_dir
        # Define model
        self.model = STLROneVAE(
            view_szes=view_szes,
            latent_sz=latent_sz,
            rnn_hidden_sz=rnn_hidden_sz,
            emission_hidden_sz=emission_hidden_sz,
            transition_hidden_sz=transition_hidden_sz,
            n_layers=n_layers,
            dropout=dropout,
            nonlinearity=nonlinearity
        )
        # self.model.apply(self.init_weights) # It will apply recursively to submodules of the model
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        # After save_hyperparamters
        self.result_auc = {}

    def init_weights(self, model):
        if isinstance(model, nn.Linear):
            torch.nn.init.xavier_normal_(model.weight.data, gain=1.41)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def run_step(self, X):
        output, loss = self.model(X)
        loss_dict = {
            'loss': loss['negELBO'].mean(),
            'kld': loss['kld'].mean().detach(),
            'logllh': loss['logllh'].mean().detach()
        }
        return loss_dict, output

    def training_step(self, batch, batch_idx):
        # Initialization
        X, _ = batch
        train_metrics = {}
        # Calculation
        loss_dict, _ = self.run_step(X)
        train_metrics["train/loss"] = loss_dict["loss"]
        train_metrics["train/kld"] = loss_dict["kld"]
        train_metrics["train/logllh"] = loss_dict["logllh"]
        self.log_dict(
            train_metrics, 
            prog_bar=False, 
            logger=False, 
            on_step=True, 
            on_epoch=False)
        return loss_dict
    
    def calc_anomaly_score(self, X, y, output):
        metrics = {}
        X = [Xv.transpose(0, 1) for Xv in X]
        zs = torch.stack(output['z_post'])
        scores = self.model.get_score(zs, X)
        for key in scores.keys():
            metrics[key] = scores[key]
        metrics['ground_truth'] = y
        metrics['X'] = X
        metrics['Z'] = zs
        return metrics
    
    def calc_sne_score(self, outputs):
        # X: V x (T, B, dv)
        # Z: (V, T, B, dz)
        V = len(self.view_szes)
        X = [torch.concat([output['X'][v] for output in outputs], dim=1) for v in range(V)]
        Z = torch.concat([output['Z'] for output in outputs], dim=2)
        X_last = [Xv[-1] for Xv in X]   # V x (N, dv)
        Z_last = Z[:,-1]   # (V, N, dz)
        # print(Z_last.size())
        # print(X_last[0].size())
        return sne_score_fn(Z_last, X_last, k=30)
    
    def evaluate_auc(self, outputs, stage):
        metrics = {}
        # Loss
        for key in ['loss', 'kld', 'logllh']:
            tmp = [output[key] for output in outputs]
            metrics[f"{stage}/{key}"] = torch.stack(tmp).mean()
        # Ground truth
        ground_truth = [output['ground_truth'] for output in outputs]
        ground_truth = torch.concat(ground_truth)
        # Calculate AUCs
        score_keys = [key for key in outputs[0].keys() if 'score' in key]
        for key in score_keys:
            score = torch.concat([output[key] for output in outputs])
            # metrics[f"auc_{key}"] = auroc(score, ground_truth)
            metrics[f"auc_{key}"] = roc_auc_score(ground_truth.cpu().numpy(), score.cpu().numpy())
        
        # # Append sne score
        # if (self.current_epoch+1) % 5 == 0:
        #     scores = self.calc_sne_score(outputs)
        #     for key, score in scores.items():
        #         metrics[f"auc_{key}"] = roc_auc_score(ground_truth.cpu().numpy(), score.numpy())
        # else:
        #     # metrics['auc_sne_1kl'] = 0.
        #     metrics['auc_sne_crossview_1kl'] = 0.
        
        # Save scores and ground truth
        save_result(outputs, ground_truth, self.output_dir)
        # Save auc
        auc_keys = [key for key in metrics.keys() if 'auc' in key]
        for key in auc_keys:
            if key in self.result_auc.keys():
                # self.result_auc[key].append(metrics[key].cpu().numpy())
                self.result_auc[key].append(metrics[key])
            else:
                # self.result_auc[key] = [metrics[key].cpu().numpy()]
                self.result_auc[key] = [metrics[key]]
        return metrics

    def validation_step(self, batch, batch_idx):
        # Initialization
        X, y = batch
        # Calculation
        loss_dict, output = self.run_step(X)
        valid_metrics = self.calc_anomaly_score(X, y, output)
        valid_metrics.update(loss_dict)
        return valid_metrics

    def validation_epoch_end(self, outputs):
        valid_metrics = self.evaluate_auc(outputs, stage='valid')
        self.log_dict(
            valid_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        print_metrics(valid_metrics)
    
    def test_step(self, batch, batch_idx):
        # Initialization
        X, y = batch
        # Calculation
        loss_dict, output = self.run_step(X)
        test_metrics = self.calc_anomaly_score(X, y, output)
        test_metrics.update(loss_dict)
        return test_metrics
    
    def test_epoch_end(self, outputs):
        test_metrics = self.evaluate_auc(outputs, stage='test')
        self.log_dict(
            test_metrics,
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=True
        )
        print_metrics(test_metrics)
    