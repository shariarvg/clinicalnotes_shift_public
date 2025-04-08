import torch

class NotesDataset(torch.utils.data.Dataset):
    def __init__(self, data, task, tokenizer, max_length):
        self.notes_data = list(data['text']) # List of tuples (original, positive, negative)
        self.labels = list(data[task])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.notes_data)

    def __getitem__(self, idx):
        text = self.notes_data[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }