torch 與 CUDA對應版本
https://pytorch.org/get-started/previous-versions/
12345

資料位置： /wk171/yungyun/EC_downscaling_code/EC_torch_unet
## Requirements
為了確保每個專案的環境互不干擾，建議開啟虛擬環境(非必要，可以不執行，但可能會影響其他沒有版控的專案)
建議使用conda也可用python3 (conda 環境之後可以run predict.ipynb)
目前是選用torch==2.1.1 和 cuba==12.1版
``` 
conda create --name EC_torch python=3.10
conda activate EC_torch
pip install -r requirements.txt 
```
離開專案，退出虛擬環境
```
deactivate
```

## Training
進入虛擬環境後: conda activate EC_torch

To run the experiments, run this command:
```train
python main.py <experiment_path>

Example:
python main.py experiments/EC_unet_downscaling_5km_OK.yml 
```
目前設定使用0號gpu
可由main.py 第10行更改
```
os.environ['CUDA_VISIBLE_DEVICES']="0"
```
#### Tensorbroad
checking the GAN model training
```
tensorboard --logdir=<log_path> --port=<four_numbers> --bind_all

for example:
tensorboard --logdir=/wk171/yungyun/EC_downscaling_code/EC_torch_unet_publish/logs --port=6526 --bind_all
```

#### for perdict
Look at the predict.ipynb


#### LCL resource
https://github.com/pytorch/pytorch/pull/1583/files