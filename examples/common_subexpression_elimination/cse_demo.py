import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        """Initializes the Model class with a single LayerNorm layer of embedding dimension 10."""
        super().__init__()
        embedding_dim = 10
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """Applies LayerNorm to the input tensor and adds it to an independently computed LayerNorm of the same
        tensor.
        """
        return self.layer_norm(x) + F.layer_norm(x, [10])


layer_norm = Model()

batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
torch.onnx.export(layer_norm, embedding, "ln_cse.onnx", opset_version=13)
