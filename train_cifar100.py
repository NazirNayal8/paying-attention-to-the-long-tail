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


config = read_config('configs/classification/cifar100_config_local.yaml')

if isinstance(config['class_names'], str):
    class_names = read_config(config['class_names'])
    config['class_names'] = list(class_names.values())

pl.seed_everything(config['random_seed'])

run_name = 'cifar100_rho10_mixup_alpha1_randaug_2-10_CE'

model = Transformer(config)
wandb_logger = WandbLogger(name=run_name, project='attention_LT', job_type='train', log_model=True)

checkpoint_callback = ModelCheckpoint(
    dirpath='model_logs/',
    monitor='val/accuracy',
    mode='max',
    filename=run_name + '-{val/accuracy:.4f}-{epoch:02d}'
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
#trainer.test(model)