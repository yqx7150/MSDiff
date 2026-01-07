# MSDiffiff

**Paper**: MSDiffiff: Multi-Scale Diffusion Model for Ultra-Sparse View CT Reconstruction
         
**Authors**: Pinhuang Tan; Mengxiao Gen; Jinya Lu; Yi Liu; Bin Huang; Qiegen Liu          


## Training
Sparse-view Diffusion Model (SDM)
```bash

CUDA_VISIBLE_DEVICES=0 python main_120.py --config=aapm_sin_ncsnpp_120.py --workdir=exp_120 --mode=train --eval_folder=result

```

Full-view Diffusion Model (FDM) 
```bash

python main_720.py --config=aapm_sin_ncsnpp_720.py --workdir=exp_720 --mode=train --eval_folder=result

```
## Test
```bash
CUDA_VISIBLE_DEVICES=0 python PCsampling_demo.py
```


## Test Data
In file './Test_CT', 12 sparse-view CT data from AAPM Challenge Data Study.
