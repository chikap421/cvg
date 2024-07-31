# Patch pkg_resources before importing pytorch_lightning
import pkgutil

if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        pass

    pkgutil.ImpImporter = ImpImporter

import importlib.machinery
importlib.machinery.FileFinder.find_module = importlib.machinery.FileFinder.find_spec

import psutil
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from videogpt import VQVAE, VideoData

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/training_data.json')
    args = parser.parse_args()

    data = VideoData(args)
    data.train_dataloader()
    data.val_dataloader()

    model = VQVAE(args)

    callbacks = [ModelCheckpoint(monitor='val/recon_loss', mode='min')]
    logger = CSVLogger("logs", name="videogpt")

    trainer_args = {
        'max_steps': args.max_steps,
        'callbacks': callbacks,
        'logger': logger,
        'accelerator': 'gpu' if args.gpus > 0 else 'cpu',
        'devices': args.gpus if args.gpus > 0 else None
    }

    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage before training: {memory_info.rss / 1024 / 1024} MB")

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()

