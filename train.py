from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import hydra


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
       print("Preparing data...")
       dataloader = hydra.utils.instantiate(cfg.dataloader_creator)
       dataloader_train, dataloader_val = dataloader.get_dataloaders()

       print("Loading the model...")
       model = hydra.utils.instantiate(cfg.model)
       trainer = pl.Trainer(**cfg.trainer)

       print("Initializing training...")
       trainer.fit(model, dataloader_train, dataloader_val)

       print("Saving the model...")
       torch.save(model.state_dict(), "/Users/ben/inz/cement/saved_models/cement_model.pkl")

      
if __name__ == "__main__":
    main()