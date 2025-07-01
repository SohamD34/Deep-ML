import numpy as np


def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):

    features = input_data.copy()
    
    N, H, W, C0 = input_data.shape
    kh, kw = kernel_size
    
    pad_h = kh // 2
    pad_w = kw // 2
    
    for layer_idx in range(num_layers):
        current_channels = features.shape[3]
        
        expected_input_channels = C0 + layer_idx * growth_rate
        if kernels[layer_idx].shape[2] != expected_input_channels:
            raise ValueError(f"Layer {layer_idx}: kernel input channels ({kernels[layer_idx].shape[2]}) "
                           f"don't match expected channels ({expected_input_channels})")
        
        activated_features = np.maximum(0, features)
        
        padded_features = np.pad(activated_features, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        
        kernel = kernels[layer_idx]
        
        conv_output = np.zeros((N, H, W, growth_rate))
        
        for n in range(N):
            for out_c in range(growth_rate):
                for h in range(H):
                    for w in range(W):
                        patch = padded_features[n, h:h+kh, w:w+kw, :]
                        conv_output[n, h, w, out_c] = np.sum(patch * kernel[:, :, :, out_c])
        
        features = np.concatenate([features, conv_output], axis=3)
    
    return features