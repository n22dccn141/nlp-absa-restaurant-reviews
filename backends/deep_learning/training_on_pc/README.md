# ABSA Deep Learning Training on Windows PC

Copy this whole `training_on_pc` folder to your RTX 3060 Windows PC.

## 1. Create a Virtual Environment

```bat
cd training_on_pc
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install PyTorch with CUDA

For RTX 3060, install the CUDA build of PyTorch:

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining packages:

```bat
pip install -r requirements.txt
```

Check CUDA:

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

It should print `True` and your RTX GPU name.

## 3. Quick Smoke Test

Run this first to make sure the code works:

```bat
python train_on_pc.py --quick
```

## 4. Full Training

Recommended command:

```bat
python train_on_pc.py --model microsoft/deberta-v3-base --epochs 5 --batch-size 8 --gradient-accumulation-steps 2
```

If CUDA runs out of memory, use:

```bat
python train_on_pc.py --model microsoft/deberta-v3-base --epochs 5 --batch-size 4 --gradient-accumulation-steps 4
```

## 5. Outputs

After training, the folder will contain:

```text
model_output/bert_absa_model/
model_output/bert_training_report.json
model_output/bert_absa_model.zip
```

Bring this back to your Mac:

```text
model_output/bert_absa_model.zip
model_output/bert_training_report.json
```

On the Mac, unzip/copy the model folder into:

```text
backends/deep_learning/bert_absa_model/
```

and copy the report to:

```text
backends/deep_learning/bert_training_report.json
```
