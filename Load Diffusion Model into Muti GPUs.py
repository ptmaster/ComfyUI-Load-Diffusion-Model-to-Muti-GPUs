import types
import torch
import comfy.model_management

# 基础的用于覆盖设备相关设置的类，定义通用逻辑
class OverrideDiffusionDevice:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu", ]
        for k in range(0, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")

        return {
            "required": {
                "device": (devices, {"default": "cpu"}),
            }
        }

    FUNCTION = "patch"
    CATEGORY = "other"

    def override(self, model, model_attr, device):
        # 设置模型以及相关修补器（patcher）的设备属性
        model.device = device
        patcher = getattr(model, "patcher", model)
        for name in ["device", "load_device", "offload_device", "current_device", "output_device"]:
            setattr(patcher, name, device)

        # 将模型中的具体模块移动到指定设备
        py_model = getattr(model, model_attr)
        py_model.to = types.MethodType(torch.nn.Module.to, py_model)
        py_model.to(device)

        # 移除模型再次移动设备的能力（可根据实际情况进一步调整逻辑）
        def to(*args, **kwargs):
            pass
        py_model.to = types.MethodType(to, py_model)
        return (model,)

    def patch(self, *args, **kwargs):
        raise NotImplementedError


# 针对Diffusion模型的设备覆盖类，连接Load Diffusion Model节点的model输出端
class OverrideLoadedDiffusionDevice(OverrideDiffusionDevice):
    @classmethod
    def INPUT_TYPES(s):
        k = super().INPUT_TYPES()
        k["required"]["diffusion_model"] = ("MODEL",)  # 这里假设Load Diffusion Model节点输出的类型名为"MODEL"，可根据实际调整
        return k

    RETURN_TYPES = ("MODEL",)
    TITLE = "Force/Set Loaded Diffusion Model Device"

    def patch(self, diffusion_model, device):
        return self.override(diffusion_model, "model", torch.device(device))


NODE_CLASS_MAPPINGS = {
    "OverrideLoadedDiffusionDevice": OverrideLoadedDiffusionDevice
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}