import wandb
import pytorch_lightning as pl
import yaml
import json
from transformer_pl import Transformer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

default_config_path = 'configs/classification/imagenet_config.yaml'
multi_experiment_config_path = 'configs/classification/multi_experiment_imagenet.yaml'
project_name = 'attention_LT_imagenet'


def read_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_json(path):
    
    with open(path, 'rb') as f:
        file_contents = json.load(f)
    return file_contents


def run_experiment(run_name, config):

    pl.seed_everything(config['random_seed'])

    if isinstance(config['class_names'], str):
        class_names = read_json(config['class_names'])
        config['class_names'] = class_names

    model = Transformer(config)
    wandb_logger = WandbLogger(name=run_name, project=project_name, job_type='train', log_model=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_logs/imagenet',
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

