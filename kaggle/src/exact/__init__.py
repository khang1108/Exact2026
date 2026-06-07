# Torch 2.2.x and legacy versions compatibility monkey-patch 
# This intercepts calls from newer versions of `accelerate` and `transformers` 
# that expect `torch.is_autocast_enabled(device_type)` to take arguments.
try:
    import torch
    if hasattr(torch, "is_autocast_enabled"):
        try:
            # Check if it accepts an argument (PyTorch >= 2.4 behavior)
            torch.is_autocast_enabled("cuda")
        except TypeError:
            # It threw a TypeError, meaning it takes no arguments (PyTorch < 2.4 behavior)
            _orig_is_autocast = torch.is_autocast_enabled
            def _patched_is_autocast_enabled(*args, **kwargs):
                return _orig_is_autocast()
            torch.is_autocast_enabled = _patched_is_autocast_enabled
except Exception:
    pass
