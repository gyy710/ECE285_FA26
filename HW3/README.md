# ECE 285 HW3 · Histopathology image synthesis

The two-page [report](output/pdf/ECE285_HW3_report.pdf) presents the trained diffusion model, loss curve, generated patches and evaluation. The [training log](results/training.csv) contains one row per update. The [evaluation output](results/metrics.json) records full-precision scores and the preprocessing used.

## Generate new images from the saved weights

The experiment used Python 3.11 and PyTorch with MPS on an Apple M4 Max. The scripts also accept CPU or CUDA. Install the packages in [requirements.txt](requirements.txt) in a virtual environment. Run the following from the HW3 directory to produce 32 new PNGs in a fresh folder.

```bash
python sample.py --checkpoint results/checkpoint.pt --out new_samples --count 32 --seed 42
```

To reproduce the submitted sample set, use seed 286, count 518, batch size 32 and 48 sampling steps. The [checkpoint](results/checkpoint.pt) contains the EMA denoiser weights and training configuration. The [saved sample set](results/samples) is ordered by draw index.

## Reproduce the experiment

The HW3 workspace contains the prepared image array and split. The compact submission archive includes the split and manifest. To rebuild the image array after unpacking the archive, run `python prepare_data.py` with network access, or supply `--archive /path/to/data-final.zip` for a local Kaggle v1 archive. Preparation selects the Histopathology folder and records the source path, split, numeric-stem group and resized-image hash in [the manifest](data/manifest.csv).

Run the following commands in a newly created environment with the dependencies installed. They use a new output folder and reproduce the training and sampling settings. Numerical results can vary with software or accelerator.

```bash
python train.py --out rerun --updates 20000
python sample.py --checkpoint rerun/checkpoint.pt --out rerun/samples --count 518 --seed 286
python evaluate.py --samples rerun/samples --output rerun/metrics.json
```

On the first evaluation, torchvision retrieves its ImageNet Inception-v3 weights if they are absent from the local cache. Both FIDs use 518 held-out images. The 1,000-dimensional score uses class probabilities from the Ignite tutorial. The 2,048-dimensional score uses pooled activations from the same torchvision model. The IS is a single estimate across the entire set. [Evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)

## Rebuild the report

From HW3, run `python report/make_figures.py` using the submitted files in `results/`. In `report/`, run `latexmk -pdf main.tex` to compile the LaTeX source with the supplied course NeurIPS 2025 style.
