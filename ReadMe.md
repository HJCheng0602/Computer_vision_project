# 3D Pottery Generation via GAN

## report
the final report is available [here](report/cv_report.pdf)

## setup
To set up the environment, run:
```bash
git clone 
cd VoxPottery
conda create -n VoxPottery python=3.8 -y
conda activate VoxPottery
pip install -r requirements.txt
```

## data preparation
To preprocess the raw data, look into the `preprocess.py` to transform the raw `.vox` 
files into processed numpy arrays stored in `data/processed_data64/`.

**Attention** : because we have plotted a $64^3$ version, the $32^3$ version is no longer supported in this readme, but you still can read the code in `preprocess.py` to generate $32^3$ data if needed.

## training
To train the GAN model, run:
```bash
bash scripts/train_64.sh
```

by default, you will train a GAN model using our last parameter setting which we reported in our final report as `Z64_Glr0.002_Dlr0.0002_Res64_BS64_Resnet_SN`.

Besides, if you want to use our other checkpoints, you can find them in the [huggingface link](https://huggingface.co/HJCheng0602/cv_project).

## visualization
To visualize the generated pottery shapes, run:
```bash
bash scripts/visualization.sh
```

please be careful to set the correct `MODEL_DIR`, `OUTPUTDIR`, and `MODEL_TYPE` in the `visualization.sh` script before running it.

a simple guide sheet is here:
| MODEL_NAME | MODEL_TYPE | path to MODEL_DIR |
|-------------------------------|----------------|------------------------------|
| Init | Init | checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64/generator_epoch_100.pth |
| Wider_G | Wider_G | checkpoints/Z64_Glr0.002_Dlr0.0002_WiderG_Res64_BS64/generator_epoch_100.pth |
|Wider_GSN | Wider_GSN | checkpoints/Z64_Glr0.002_Dlr0.0002_WiderG_Res64_BS64_Wider_GSN/generator_epoch_100.pth |
| Resnet_SN | resnet_SN | checkpoints/Z64_Glr0.002_Dlr0.0002_Res64_BS64_Resnet_SN/generator_epoch_100.pth |

besides, we provide our tensorboard logs in the `logs/` folder for your reference.

**Attention** : the `checkpoints/` and `logs/` folders are not included in this repo due to the size limit. You can download them from the [huggingface link](https://huggingface.co/HJCheng0602/cv_project).

## evaluation
To evaluate the trained model, run:
```bash
bash scripts/evaluation.sh
```

like the visualization script, please be careful to set the correct `MODEL_DIR`, `OUTPUTDIR`, and `MODEL_TYPE` in the `evaluation.sh` script before running it.