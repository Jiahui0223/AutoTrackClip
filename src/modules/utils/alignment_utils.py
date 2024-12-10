import torch

def align_time_dimensions(slow, fast):
    """
    对齐 Slow 和 Fast 通道的时间维度。
    Args:
        slow (torch.Tensor): Slow Pathway 张量，形状 [B, C, T_slow, H, W]。
        fast (torch.Tensor): Fast Pathway 张量，形状 [B, C, T_fast, H, W]。
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 对齐后的张量。
    """
    T_slow = slow.shape[2]
    T_fast = fast.shape[2]

    if T_slow != T_fast:
        if T_slow < T_fast:
            # 子采样 Fast Pathway
            indices = torch.linspace(0, T_fast - 1, T_slow).long()
            fast = fast[:, :, indices, :, :]
        else:
            # 子采样 Slow Pathway
            indices = torch.linspace(0, T_slow - 1, T_fast).long()
            slow = slow[:, :, indices, :, :]

    return slow, fast
