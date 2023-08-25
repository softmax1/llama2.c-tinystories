from re import search
from collections import defaultdict
from functools import partial
from json import dump
from pathlib import Path
from typing import List, Dict, Any, DefaultDict, Tuple, Optional

from torch import Tensor, mean, pow, randn
from torch.nn import Module
import torch

@torch.no_grad()
def kurtosis(x: Tensor, dim: int = 0) -> Tensor:
		"""
		Batch kurtosis along batch dim.
		kurtosis[x] = E[y^2] / (E[y])^2, where y = (x - E[x])^2
		excess_kurtosis = kurtosis - 3
		"""
		# Flatten input x except batch_size dim
		x = x.view(x.shape[dim], -1)
		y = pow(x - mean(x, dim=1, keepdim=True), 2)
		excess_kurtosis = mean(pow(y, 2), dim=1) / pow(mean(y, dim=1), 2) - 3.
		return excess_kurtosis

@torch.no_grad()
def compute_avg_and_std(array: List[float]) -> Dict[str, float]:
		avg = sum(array) / len(array)
		second_moment = sum([x**2 for x in array]) / len(array)
		std = (second_moment - avg**2)**0.5
		return {'avg': avg, 'std': std}


# I/O
def save_results(results: Dict[str, Dict[str, Any]], output_dir: str):
		results_dir = Path.cwd() / "results"
		results_dir.mkdir(exist_ok=True)

		model_name = output_dir.split("/")[1]
		filepath = results_dir / f"{model_name}.json"

		with filepath.open(mode='w') as fp:
				dump(results, fp, indent=4)

def flatten_dict(input_dict, parent_key='', separator='/'):
    flattened_dict = {}
    
    for key, value in input_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            flattened_dict.update(flatten_dict(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    
    return flattened_dict


# Activation Kurtosis Hook
@torch.no_grad()
def save_activations_kurtosis(
				activations: DefaultDict,
				name: str,
				module: Module,
				inp: Tuple,
				out: Tensor
) -> None:
		"""
		PyTorch Forward hook to compute moving average of kurtosis  at each forward pass.
		Mutates specified dict objects with each fwd pass.
		"""
		m = out.shape[0]
		num_params = torch.tensor(out.shape[1:]).prod().item() # exclude batch_size
		k = kurtosis(out.detach()).sum().item()
		n, mu, _ = activations[name]
		activations[name] = [n + m, (n * mu + k) / (n + m), num_params] # TODO: Weight mean by num params


def activation_hooks(
				model: Module,
				layers_to_save: Optional[List[str]] = None
) -> DefaultDict[str, List[float]]:
		"""Registers forward hooks in specified layers.
		Parameters
		----------
		model:
				PyTorch model
		layers_to_save:
				Module names within ``model`` whose activations we want to save. If None, save all layers

		Returns
		-------
		activations_dict:
				dict of lists containing activations of specified layers in
				``layers_to_save``.
		"""
		activations_dict = defaultdict(lambda: [0, 0., 0])
		hooks = []

		for name, module in model.named_modules():
				if layers_to_save is None or name in layers_to_save:
						unwanted_prefix = '_orig_mod.'
						hooks += [module.register_forward_hook(
								partial(save_activations_kurtosis, activations_dict, name.replace(unwanted_prefix, ''))
						)]
				
		def deregister():
				for hook in hooks:
						hook.remove()

		return activations_dict, deregister

CAPTURE_PATTERN = r"feed_forward\.w\d+"


def summarise_layers(activations_dict: DefaultDict[str, float], num_params: List[int]) -> Dict[str, Dict[str, float]]:
		"""
		Compute average and std of activations per layer.
		"""
		layers = defaultdict(list) # aggregate common layers over blocks
		for name, kurtosis in activations_dict.items():
				if match := search(CAPTURE_PATTERN, name):
						layers[match.group()].append(kurtosis)
		summary_per_layer = {name: compute_avg_and_std(kurtoses) for name, kurtoses in layers.items()}

		mean = sum(a*b for a,b in zip(activations_dict.values(), num_params)) / len(activations_dict) / sum(num_params)
		return {'layers': activations_dict, 'summary': summary_per_layer, 'mean': mean}

# Analysis
def kurtosis_metrics(saved_activation_kurtosis: Dict[str, List[float]], model: Module) -> dict:
		"""
		Compute weight and activation kurtosis post-inference.
		"""
		activation_kurtosis = {name: val[1] for name, val in saved_activation_kurtosis.items()}
		activation_params = [val[2] for val in saved_activation_kurtosis.values()]
		weight_kurtosis = {name.replace('_orig_mod.', ''): kurtosis(param.unsqueeze(0).detach()).item() for name, param in model.named_parameters()}
		weight_params = [param.numel() for param in model.parameters()]

		return {'activations': summarise_layers(activation_kurtosis, activation_params), 'weights': summarise_layers(weight_kurtosis, weight_params)}