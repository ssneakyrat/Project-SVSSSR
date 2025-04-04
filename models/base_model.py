import os
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple, List, Union

class BaseModule(nn.Module, ABC):
    """Base class for all model modules."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._build_model()
        
    @abstractmethod
    def _build_model(self):
        """Build the model architecture. To be implemented by subclasses."""
        pass
    
    def get_param_count(self) -> int:
        """Return the number of parameters in the module."""
        return sum(p.numel() for p in self.parameters())
    
    def print_architecture(self, name: str = None):
        """Print the module architecture."""
        name = name or self.__class__.__name__
        print(f"\n{name} Architecture:")
        print(f"  Parameters: {self.get_param_count():,}")
        
        # Print the model's structure
        children = list(self.named_children())
        if not children:
            print("  No children modules.")
        else:
            for name, child in children:
                params = sum(p.numel() for p in child.parameters())
                print(f"  • {name}: {child.__class__.__name__} ({params:,} params)")


class BaseSVSModel(pl.LightningModule, ABC):
    """Base Lightning module for SVS models with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Set up additional attributes
        self.global_step_val = 0
        self.train_steps = 0
        self.mode = config["training"]["mode"]
        
        # Initialize metrics
        self.metrics = {}
        
    def configure_optimizers(self):
        """Configure optimizers based on the config."""
        opt_config = self.config["training"]["optimizer"]
        sched_config = self.config["training"]["scheduler"]
        
        # Get model parameters
        params = list(self.parameters())
        
        # Create optimizer
        if opt_config["name"] == "adam":
            optimizer = torch.optim.Adam(
                params, 
                lr=opt_config["learning_rate"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"]
            )
        elif opt_config["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                params, 
                lr=opt_config["learning_rate"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                weight_decay=opt_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        
        # Create scheduler
        if sched_config["name"] == "none":
            return optimizer
        
        if sched_config["name"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=sched_config["decay_steps"],
                eta_min=opt_config["learning_rate"] * sched_config["min_lr_ratio"]
            )
        elif sched_config["name"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config["decay_steps"] // 3,  # 3 steps over the training
                gamma=sched_config["min_lr_ratio"] ** (1/3)
            )
        elif sched_config["name"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=sched_config["decay_steps"] // 10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['name']}")
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def get_progress_bar_dict(self):
        """Customize the progress bar information."""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        
        # Add mode to the progress bar
        items["mode"] = self.mode
        
        # Add learning rate
        if hasattr(self, "trainer") and hasattr(self.trainer, "optimizers"):
            if len(self.trainer.optimizers) > 0:
                items["lr"] = f"{self.trainer.optimizers[0].param_groups[0]['lr']:.1e}"
        
        return items
    
    def log_dict_with_prefix(self, metrics: Dict[str, float], prefix: str = "", prog_bar: bool = False):
        """Log a dictionary of metrics with an optional prefix."""
        prefixed_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
        self.log_dict(prefixed_metrics, prog_bar=prog_bar)
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Validation step. To be implemented by subclasses."""
        pass
    
    def on_train_batch_start(self, batch, batch_idx):
        """Called before each training batch."""
        self.train_steps += 1
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_validation_start(self):
        """Called at the beginning of validation."""
        self.global_step_val = self.global_step
        return super().on_validation_start()
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file with interpolation support."""
        # Load config with OmegaConf for variable interpolation
        config = OmegaConf.load(config_path)
        
        # Create proper scaling operations
        def multiply(a, b):
            return a * b
        
        def divide(a, b):
            return a // b
        
        def add(a, b):
            return a + b
        
        # Register operations
        OmegaConf.register_resolver("mul", multiply)
        OmegaConf.register_resolver("div", divide)
        OmegaConf.register_resolver("add", add)
        
        # Convert to dict
        return OmegaConf.to_container(config)