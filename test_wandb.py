import wandb
wandb.init(project="test-project")
wandb.log({"test": 1})
wandb.finish()