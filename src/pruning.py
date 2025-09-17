import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model, conv_prune_amount=0.2, linear_prune_amount=0.1):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=conv_prune_amount)
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)
 

def apply_pruning(model):
    """Removes pruning reparameterization from a model permanently.
    This allows the model to be saved/loaded normally without pruning artifacts.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Check if the module was pruned
            if prune.is_pruned(module):
                 
                prune.remove(module, 'weight')
 
                if hasattr(module, 'weight_orig'):
                    del module.weight_orig
                if hasattr(module, 'weight_mask'):
                    del module.weight_mask
                
                # Ensure weight is now a regular parameter
                assert not prune.is_pruned(module)
    return model