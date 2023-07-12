import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ALIKE_code.nets.alnet_dkd import ALNet
from pytorch_lightning.plugins.io import TorchCheckpointIO as tcio
from PIL import Image
from torchvision import transforms


def get_model(checkpoint_path,device):
    model = ALNet(c1=16, c2=32, c3=64, c4=128, dim=128, agg_mode='cat', single_head=True)
    model.load_from_checkpoint(checkpoint_path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model



def main(checkpoint_path,OUTPUT_MODEL,device):
    image_path = "/media/xin/work1/github_pro/ALIKE/ALIKE_code/model_transfor/img/1.png"
    # 加载图片
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 将PIL类型转化成numpy类型
    image = np.array(image)
    transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    # 加载模型
    model = get_model(checkpoint_path,device)
    model_res = model(image)  # 模型测试
    # 直接将模型保存为pth
    torch.save(model.state_dict(),OUTPUT_MODEL)
    state_dict = torch.load(OUTPUT_MODEL)
    model.load_state_dict(state_dict)
    model_res2 = model(image)
    print("done")

    # 检查结果
    print(torch.equal(model_res[0], model_res2[0]))
    print(torch.equal(model_res[1], model_res2[1]))



if __name__ == "__main__":
    ckpt_path = '/media/xin/work1/github_pro/ALIKE/test_model/R4.0.0/epoch=125-mean_metric=0.3589.ckpt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_model = "125_torch_model.pth"
    main(ckpt_path,output_model,device)
