from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import hydra


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
       print("Loading the model...")
       model = hydra.utils.instantiate(cfg.model)

       print("Loading model's weights ...")
       model.load_state_dict(torch.load("/Users/ben/inz/cement/saved_models/cement_model.pkl"))
       model.eval()

       print("Making prediction...")
       test_sample = torch.tensor([-0.61207995, -0.26720917,  0.56516602, -1.00230708])
       model = model.eval()
       testowy_wynik = round(model(test_sample).detach().numpy()[0])
       print(f"Predicted compressive strength is: {testowy_wynik} kPa")

if __name__ == "__main__":
    main()