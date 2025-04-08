from typing import Any, Dict, List, Optional, Union
import torch
from torch import nn
from kronfluence.task import Task

criterion = nn.CrossEntropyLoss()

class CrossEntropyTask(Task):
        
    def compute_train_loss(
        self,
        batch: Any,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        loss = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], labels = batch['labels'])['loss']
        return loss

    def compute_measurement(
        self,
        batch: Any,
        model: nn.Module,
    ) -> torch.Tensor:
        # TODO: Complete this method.
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        
        return ['classifier']
    
    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        # TODO: [Optional] Complete this method.
        return None  # Attention mask not used.