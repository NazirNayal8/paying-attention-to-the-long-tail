import wandb
import pytorch_lightning as pl
import yaml
from transformer_pl import Transformer
from pytorch_lightning.loggers import WandbLogger


def read_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = read_config('configs/cifar10_config_local.yaml')

pl.seed_everything(config['random_seed'])

wandb.init(config=config)

config['learning_rate'] = wandb.config.learning_rate
config['weight_decay'] = wandb.config.weight_decay

model = Transformer(config)
wandb_logger = WandbLogger(name=f'hyper_opt_{wandb.config.learning_rate}', job_type='train', log_model=True)

trainer = pl.Trainer(
    max_epochs=wandb.config.num_epochs,
    enable_checkpointing=True,
    fast_dev_run=False,
    overfit_batches=False,
    gpus=1,
    logger=wandb_logger,
    log_every_n_steps=1,
    profiler='simple',
)
trainer.fit(model)
trainer.test(model)


