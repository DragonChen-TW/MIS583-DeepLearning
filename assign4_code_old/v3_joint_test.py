import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from v3_models import get_joint_resnet, get_joint_resnext101
from v3_dataset import get_deep_fashion

batch_size = 126
workers = 16

if __name__ == '__main__':
    device = torch.device('cuda:1')
    model = get_joint_resnext101().to(device)

    ckpt = 'saves/res101_1e-3_3e-4_e20/ckpts/e004.pt'
    model.load_state_dict(torch.load(ckpt))

    _, _, dataset = get_deep_fashion()
    data = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

    total_out_cate = []
    total_out_attr = []

    model.eval()
    loader = tqdm(enumerate(data), ascii=True, total=len(data))
    for i, image in loader:
        image = image.to(device)

        out_cate, out_attr = model(image)

        pred_cate = torch.argmax(out_cate, dim=1)
        total_out_cate.append(pred_cate)

        pred_attr = (torch.sigmoid(out_attr) > 0.5).int()
        total_out_attr.append(pred_attr)

        # if i == 2:
        #     break
    
    total_pred_cate = torch.cat(total_out_cate).cpu().numpy().tolist()
    # print(total_pred_cate)
    total_pred_attr = torch.cat(total_out_attr).cpu().numpy().tolist()
    # print(total_pred_attr)

    with open('pred_cate.csv', 'w') as f:
        f.write('file_path,category_label\n')
        for i in range(len(total_pred_cate)):
            f.write('{},{}\n'.format(dataset.img_list[i], total_pred_cate[i]))
    
    with open('pred_attr.csv', 'w') as f:
        f.write('file_path,attribute_label\n')
        for i in range(len(total_pred_attr)):
            s = dataset.img_list[i] + ','
            lbls = [n1 * n2 for n1, n2 in zip(total_pred_attr[i], range(15))]
            lbls = [str(l) for l in lbls if l != 0]
            s += ' '.join(lbls)
            s += '\n'
            f.write(s)
        