import torch
import torch.nn.functional as F
from torch.nn import Module
from src.modules.utils.alignment_utils import align_time_dimensions

class DebugSlowFast(Module):
    def __init__(self, slowfast_model):
        """
        DebugSlowFast 包装器，用于调试 SlowFast 模型的各层输出。
        Args:
            slowfast_model (torch.nn.Module): SlowFast 模型。
        """
        super(DebugSlowFast, self).__init__()
        self.slowfast_model = slowfast_model

    def forward(self, inputs):
        slow_pathway, fast_pathway = inputs
        print(f"Input Slow Pathway shape: {slow_pathway.shape}")
        print(f"Input Fast Pathway shape: {fast_pathway.shape}")

        # 遍历模型的每一层
        for name, module in self.slowfast_model.named_children():
            try:
                # 对于子模块列表（如 Stage），逐个处理
                if isinstance(module, torch.nn.ModuleList):
                    for idx, submodule in enumerate(module):
                        slow_pathway, fast_pathway = submodule([slow_pathway, fast_pathway])
                        print(f"Submodule {name}[{idx}]: Slow={slow_pathway.shape}, Fast={fast_pathway.shape}")
                else:
                    # 其他普通模块
                    slow_pathway, fast_pathway = module([slow_pathway, fast_pathway])
                    print(f"Layer {name}: Slow={slow_pathway.shape}, Fast={fast_pathway.shape}")
            except Exception as e:
                print(f"Error in {name}: {e}")
                break

        # 确保时间维度对齐
        slow_pathway, fast_pathway = align_time_dimensions(slow_pathway, fast_pathway)
        print(f"Aligned Slow Pathway shape: {slow_pathway.shape}")
        print(f"Aligned Fast Pathway shape: {fast_pathway.shape}")

        return torch.cat([slow_pathway, fast_pathway], dim=1)
