import argparse
from pathlib import Path
import omegaconf
import torch

from token_topicer.cross_encoder.model import CrossEncoderModel

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, required=True, help="Path to the model file.")
    parser.add_argument("--config", type=Path, required=False, help="Path to the model config file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the traced model.")
    parser.add_argument("--map-location", type=str, default="cpu", help="Map location for loading the model.")

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = omegaconf.OmegaConf.load(f)
    
    model_checkpoint = torch.load(args.model)

    # remove .model prefix from state dict keys
    new_state_dict = {}
    for k, v in model_checkpoint["state_dict"].items():
        if k.startswith("model."):
            new_key = k[len("model.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v
    model_checkpoint["state_dict"] = new_state_dict

    model = CrossEncoderModel(
        lm_path=config["model"]["lm_path"],
        output_projection_layers=config["model"]["output_projection_layers"],
    )
    model.load_state_dict(model_checkpoint["state_dict"])
    model.to(args.map_location)
    model.eval()

    example_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long, device=args.map_location)
    example_attention_mask = torch.ones((1, 128), dtype=torch.long, device=args.map_location)

    # scripted_model = torch.jit.script(model)
    scripted_model = torch.jit.trace(
        model, (example_input_ids, example_attention_mask)
    )
    scripted_model.save(args.output)

    reloaded_model = torch.jit.load(args.output, map_location=args.map_location)
    reloaded_model.eval()

    check_input_ids = torch.randint(0, 1000, (2, 128), dtype=torch.long, device=args.map_location)
    check_attention_mask = torch.ones((2, 128), dtype=torch.long, device=args.map_location)

    with torch.no_grad():   
        model_outputs = model(
            input_ids=check_input_ids,
            attention_mask=check_attention_mask,
        )

    with torch.no_grad():
        reloaded_outputs = reloaded_model(
            check_input_ids,
            check_attention_mask,
        )

    print(torch.allclose(
        model_outputs, reloaded_outputs, atol=1e-6
    ))


if __name__ == "__main__":
    main()
