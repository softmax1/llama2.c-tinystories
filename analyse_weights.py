from argparse import ArgumentParser
import torch
from model import ModelArgs, Transformer
import json

parser = ArgumentParser()

parser.add_argument("--out-dir", required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    # resume training from a checkpoint.
    out_dir = args.out_dir
    ckpt_path = f"{out_dir}/ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_args = checkpoint["model_args"]
    # create the model
    gptconf = ModelArgs(**checkpoint_model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    weights_dist = {}
    all_weights = []
    for name, param in model.named_parameters():
        mean = param.data.mean()
        diffs = param.data - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0

        outliers = torch.logical_or(
            param.data > mean + 6 * std, param.data < mean - 6 * std
        ).sum()

        weights_dist[name] = {
            "mean": mean.item(),
            "var": var.item(),
            "std": std.item(),
            "skews": skews.item(),
            "kurtosis": kurtosis.item(),
            "outliers": outliers.item(),
            "outlier_percent": outliers.item() / param.data.numel(),
        }

        all_weights.append(param.data.detach().flatten(0).cpu().numpy())

    results_path = f"{out_dir}/weights.json"
    with open(results_path, "w") as f:
        json.dump(weights_dist, f, indent=2)
