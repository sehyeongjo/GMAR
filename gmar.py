import torch
import torch.nn.functional as F
import numpy as np  # NumPy for numerical operations

class GMAR:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.activations = None
        self.handlers = []  # List to keep track of hooks
        self.target_layer = target_layer
        self._register_hooks()  # Register hooks to the target layer

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad[0].detach())  # Store and reshape the output gradients        

    def _register_hooks(self):
        self.handlers.append(self.target_layer.register_forward_hook(self._get_features_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(self._get_grads_hook))

    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)  # Rearrange dimensions to (B, C, H, W)
        return result

    def weight_generate(self, inputs, type="l1"):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs, output_attentions=True)  # Forward pass

        index = output.logits.argmax(dim=1).item()  # Get the index of the highest score
        target = output.logits[0, index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        gradient = self.gradient[0].detach().cpu().numpy()

        num_heads = 16
        head_gradients = np.split(gradient, num_heads, axis=0)  # Split by channels

        l1_arr = []
        l2_arr = []

        for i, head_gradient in enumerate(head_gradients):

            l1_norm = np.sum(np.abs(head_gradient))
            l1_arr.append(l1_norm)

            # L2 norm calculation
            l2_norm = np.sqrt(np.sum(head_gradient**2))
            l2_arr.append(l2_norm)
        
        w_h_l1 = [l1 / sum(l1_arr) for l1 in l1_arr]
        w_h_l2 = [l2 / sum(l2_arr) for l2 in l2_arr]
        
        if type == "l1":
            return w_h_l1
        else:
            return w_h_l2    
    
    def clean_up(self):
        for handle in self.handlers:
            handle.remove()

    def attention_rollout_with_head_weights(self, attention_matrices, head_weights, residual_ratio, device):
        assert len(attention_matrices) > 0, "attention_matrices list is empty."

        num_tokens = attention_matrices[0].shape[-1]
        batch_size = attention_matrices[0].shape[0]
        num_heads = attention_matrices[0].shape[1]

        rollout_matrix = torch.eye(num_tokens, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_heads, 1, 1)

        for layer_attention in attention_matrices:
            assert layer_attention.shape[1] == len(head_weights), "Mismatch between heads and head weights."        
            weights_tensor = torch.tensor(head_weights, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            weighted_attention = layer_attention * weights_tensor

            # Update rollout matrix
            rollout_matrix = torch.matmul(rollout_matrix, weighted_attention)
            mean_value = rollout_matrix.mean() * 5e-5
            identity_matrix = torch.eye(num_tokens, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            identity_matrix = mean_value * identity_matrix.expand(batch_size, num_heads, num_tokens, num_tokens)

            rollout_matrix = rollout_matrix + (residual_ratio * identity_matrix)

        return rollout_matrix

    def compute_attention_rollout_with_head_weights(self, image_tensor, head_weights, residual_ratio=0.25):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            p = self.model(image_tensor, output_attentions=True)
            attentions = p.attentions

        attention_layers = []
        for attention_layer in attentions:
            attention_layers.append(attention_layer.detach())

        mhgr_rollout = self.attention_rollout_with_head_weights(attention_layers, head_weights, residual_ratio, device)[:, 0, 1:]
        mhgr_rollout = F.interpolate(mhgr_rollout.view(-1, 1, 14, 14), size=(224, 224), mode='bicubic')

        # Average across the layers, squeeze dimensions, and normalize
        mhgr_rollout = mhgr_rollout.mean(dim=0).squeeze()
        mhgr_rollout = (mhgr_rollout - mhgr_rollout.min()) / (mhgr_rollout.max() - mhgr_rollout.min())

        return mhgr_rollout
