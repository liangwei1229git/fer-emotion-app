import torch

model_path = "Models/Swin_Transformer/Swin_Transformer_best_model.pth"

checkpoint = torch.load(model_path, map_location='cpu')

print("Model type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys in checkpoint:")
    for i, key in enumerate(checkpoint.keys()):
        if i < 10:
            print(f"  {key}")
        if i == 10:
            print(f"  ... and {len(checkpoint.keys()) - 10} more keys")
    
    print(f"\nTotal keys: {len(checkpoint.keys())}")
elif isinstance(checkpoint, torch.nn.Module):
    print("\nModel is a torch.nn.Module")
    print(checkpoint)
else:
    print("\nUnknown model type")
    print(checkpoint)

