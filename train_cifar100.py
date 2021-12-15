import wandb
import pytorch_lightning as pl
import yaml
from transformer_pl import Transformer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def read_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = read_config('configs/cifar100_config_local.yaml')

if isinstance(config['class_names'], str):
    class_names = read_config(config['class_names'])
    config['class_names'] = list(class_names.values())

pl.seed_everything(config['random_seed'])

run_name = 'cifar100_rho_100'

model = Transformer(config)
wandb_logger = WandbLogger(name=run_name, project='attention_long_tail', job_type='train', log_model=True)

checkpoint_callback = ModelCheckpoint(
    dirpath='model_logs/',
    monitor='val/loss',
    mode='max',
    filename=run_name + '-{epoch:02d}-val_loss{val/loss:.2f}'
)

trainer = pl.Trainer(
    max_epochs=config['num_epochs'],
    enable_checkpointing=True,
    fast_dev_run=False,
    overfit_batches=False,
    gpus=1,
    logger=wandb_logger,
    log_every_n_steps=1,
    profiler='simple',
    callbacks=[checkpoint_callback]
)
trainer.fit(model)
trainer.test(model)