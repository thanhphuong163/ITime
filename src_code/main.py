import argparse
import time
import os
import pandas as pd
import torch as torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from litModule import LitModule
from datasets import *

# Setting the seed
# pl.seed_everything(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print(f"Pytorch version: {torch.__version__}")
print(f"Pytorch lightning version: {pl.__version__}")
torch.set_default_dtype(torch.float32)
n_CPU = os.cpu_count()
print(f"Number of CPUs: {n_CPU}")
n_workers = n_CPU - 6


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Dataset directory", type=str, required=True)
parser.add_argument("--output_dir", help="Output directory", type=str, required=True)
parser.add_argument("--anomaly_rate", help="Anomaly rate", type=str, required=True)
parser.add_argument("--window_size", help="Window size", type=int, required=True)
parser.add_argument("--max_epochs", help="Number of epochs", type=int, required=True)
parser.add_argument("--batch_sz", help="Batch size", type=int, required=True)
parser.add_argument("--lr", help="Learning rate", type=float, required=True)
parser.add_argument(
    "--latent_size", help="Latent size", type=int, required=True, default=20
)
parser.add_argument(
    "--rnn_hidden_size", help="RNN hidden size", type=int, required=True, default=200
)
parser.add_argument(
    "--emission_hidden_size",
    help="Emission hidden size",
    type=int,
    required=True,
    default=100,
)
parser.add_argument(
    "--transition_hidden_size",
    help="Transition hidden size",
    type=int,
    required=True,
    default=100,
)


def main(args):
    enable_logging = False
    
    # Setting hyperparameters
    data_dir = args.data_dir
    output_dir = args.output_dir
    log_dir = f"{output_dir}/logs"

    # Model hyperparameters
    latent_sz = args.latent_size
    rnn_hidden_sz = args.rnn_hidden_size
    emission_hidden_sz = args.emission_hidden_size
    transition_hidden_sz = args.transition_hidden_size

    # Training hyperparameters
    max_epochs = args.max_epochs
    batch_sz = args.batch_sz
    window_size = args.window_size
    lr = args.lr

    # Load datasets
    train_dataset = MyDataset(data_dir, "fit", window_size)
    test_dataset = MyDataset(data_dir, "test", window_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_sz, num_workers=n_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_sz, num_workers=n_workers, shuffle=False
    )
    print(f"Train size: {train_dataset.__len__()}")
    print(f"Train view dimensions: {train_dataset.view_szes}")
    print(f"Test size: {test_dataset.__len__()}")
    print(f"Test view dimensions: {test_dataset.view_szes}")

    # Declare model, trainer and logger
    model = LitModule(
        view_szes=train_dataset.view_szes,
        latent_sz=latent_sz,
        rnn_hidden_sz=rnn_hidden_sz,
        emission_hidden_sz=emission_hidden_sz,
        transition_hidden_sz=transition_hidden_sz,
        n_layers=2,
        dropout=0.0,
        nonlinearity="relu",
        lr=lr,
        output_dir=output_dir,
    )
    modelSummary = ModelSummary(max_depth=2)
    model_checkpoint = ModelCheckpoint(
        monitor="auc_logllh_oneVAE_score",
        mode="max",
        every_n_epochs=1,
        dirpath=f"{output_dir}/checkpoints",
        filename="ST-LR_{epoch:02d}_{auc_logllh_oneVAE_score:.5f}_{auc_variance_score:.5f}",
    )
    callbacks = [model_checkpoint, modelSummary] if enable_logging else [modelSummary]

    trainer = Trainer(
        gpus=1,
        deterministic=False,
        fast_dev_run=False,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,  # -1 means to check all validation data
        enable_progress_bar=True,
        enable_checkpointing=True if enable_logging else False,
        log_every_n_steps=15,
        callbacks=callbacks,
    )

    # Train model and evaluate it at every epoch
    start_time  = time.time()
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )
    running_time = time.time() - start_time

    # Save result auc
    result_df = pd.DataFrame(model.result_auc)
    result_df.to_csv(f"{output_dir}/auc.csv")
    with open(f"{output_dir}/running_time.txt", "w") as f:
        f.write(f"training time: {running_time}\n")
        print(f"total training time: {running_time} second(s)\n")


if __name__ == "__main__":
    # Run experiment
    args = parser.parse_args()
    main(args)
