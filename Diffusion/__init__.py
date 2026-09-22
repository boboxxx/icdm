"""Inference imports do not require training-only dependencies."""
from .Diffusion import ICDMSampler, ICDMTrainer


def __getattr__(name):
    if name in {"train_ICDM", "eval_JSCC_with_ICDM", "Generate_ICDM", "test_mem_and_comp", "test_mem_and_comp_of_encoder", "test_mem_and_comp_of_decoder"}:
        from importlib import import_module
        return getattr(import_module("Diffusion.Train"), name)
    raise AttributeError(name)
