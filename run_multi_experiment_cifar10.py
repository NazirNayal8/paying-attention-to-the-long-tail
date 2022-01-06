import wandb
import pytorch_lightning as pl
import yaml
from transformer_pl import Transformer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

default_config_path = 'configs/classification/cifar10_config_local.yaml'
multi_experiment_config_path = 'configs/classification/multi_experiment_cifar10.yaml'
project_name = 'attention_LT'


def read_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def run_experiment(run_name, config):

    if isinstance(config['class_names'], str):
        class_names = read_config(config['class_names'])
        config['class_names'] = list(class_names.values())

    model = Transformer(config)
    wandb_logger = WandbLogger(name=run_name, project=project_name, job_type='train', log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_logs/cifar10',
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
    wandb.finish()


def start_experiments():

    multi_experiment_config = read_config(multi_experiment_config_path)
    num_experiments = len(multi_experiment_config)

    for config in multi_experiment_config:

        run_name = list(config.keys())[0]
        experiment_config = read_config(default_config_path)

        for (key, value) in config[run_name].items():
            experiment_config[key] = value

        run_experiment(run_name, experiment_config)


start_experiments()

