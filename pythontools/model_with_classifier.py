import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ModelWithClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ModelWithClassifier, self).__init__()
        if isinstance(base_model, str):
            self.base_model = AutoModel.from_pretrained(base_model)
        else:
            self.base_model = base_model
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)  # Use correct hidden size
        self.config =self. base_model.config  # Inherit base model's config
        self.num_labels = num_labels  # Store number of labels

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None,
        output_attentions=None, output_hidden_states=None, return_dict=None
    ):
        """Handles optional arguments for compatibility with Hugging Face's Trainer"""
        if inputs_embeds is not None:
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )

        pooled_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        logits = self.classifier(pooled_output)  # Pass through classifier

        loss = None
        if labels is not None:
            if self.num_labels == 1:  # Regression task
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:  # Classification task
                loss = F.cross_entropy(logits, labels)

        # Return outputs as a dictionary
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "attentions": outputs.attentions if output_attentions else None
        }
