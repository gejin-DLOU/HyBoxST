# HyBoxST

HyBoxST predicts spatial gene expression from histology images using spot- and niche-level representations, hierarchical hyperbolic alignment, and pathway-guided box modeling. It uses a UNI image encoder with LoRA adaptation.

| Model name | Description |
| --- | --- |
| `hyboxst` | Full model with pathway-guided decoding, molecular ontology boxes, and query-box prediction. |

## 🛠️ Installation

Run all commands from the `HyBoxST_main` directory. The supplied environment targets Linux with an NVIDIA CUDA GPU; the training entry point selects a CUDA device.

```bash
conda env create -f environment.yml
conda activate HyBoxST
```

The environment specifies Python 3.12, PyTorch 2.3.1, torchvision 0.18.1, CUDA 12.1 PyTorch binaries, and FlashAttention. It includes platform-specific package builds and an absolute `prefix`; adjust the prefix for your installation if needed.

## 📥 Data Preparation

### HEST data

Use [dataset_download_hest1k.ipynb](dataset_download_hest1k.ipynb) with [HEST_v1_1_0.csv](HEST_v1_1_0.csv) to prepare the download. The notebook contains sections for kidney, colorectum, skin, and lung. The bundled [hest.py](hest.py) provides a local dataset-loader implementation.

### Processed inputs

Training requires preprocessed expression matrices and image patches. Downloading raw data alone does not produce these inputs. The current tree does not include a preprocessing entry point; prepare the following files before training:

```text
hest1k_datasets/
└── kidney/                              # Or colorectum, skin, lung
    ├── wsis/                            # Raw whole-slide images
    ├── st/                              # Raw spatial transcriptomics data
    └── processed_data/
        ├── all_slide_lst.txt
        ├── selected_gene_list.txt
        ├── selected_hvg_gene_list.txt    # Only needed for HVG runs
        ├── <slide>_filter.h5ad
        ├── spot/
        │   └── patches/<slide>_patch.npy
        └── niche/
            ├── idx/<slide>_idx.npy
            ├── patches/<slide>_patch.npy
            └── neighbors_gene_mean/<slide>_neighbors_gene_mean.npy
```

- List one slide identifier per line in `all_slide_lst.txt` and one gene per line in the selected gene list.
- Neighbor-expression columns must follow the gene order in `<slide>_filter.h5ad`.
- Spot patches are indexed by the original spot indices. Niche patches and neighbor-expression rows must align with the niche-index array.
- The loader removes spots whose selected expression values are all missing or all zero, then applies `log2(x + 1)` to spot and niche expression. Prepare expression inputs accordingly.

### Pathway resources

Resources for all four datasets are included:

```text
pathway_resources/kidney/
├── gene_pathway_mask.npy
├── selected_gene_list.txt
├── pathway_list.txt
├── pathway_genes.json
└── coverage_summary.json
```

For `hyboxst`, the mask must have shape `(number_of_pathways, number_of_genes)`. Its columns and the resource gene list must match the training gene list exactly, including order. The default resource location is `./pathway_resources/<dataset>`; override it with `--pathway_resource_dir`.

When changing the gene selection, including switching from HMHVG to HVG, prepare matching pathway resources. The current tree does not include a resource-generation script. The `hyboxst_base` variant does not require pathway resources.

## 🏋️ UNI Weights

By default, the encoder looks for:

```text
uni_weight/
└── uni/
    └── pytorch_model.bin
```

If the weights are absent, the loader attempts to download them. Provide a Hugging Face token with access to the UNI weights:

```bash
export HF_TOKEN="your_huggingface_token"
```

Pass the token to `main.py` with `--huggingface_token "$HF_TOKEN"`. For existing local weights, use `--uni_weight_path ./uni_weight`. This argument points to the parent directory containing `uni/`, not to the weight file.

## 🧩 Dataset Splits

Existing split files are provided under `split/<dataset>/flod_5/`. Each JSON contains `train_sample`, `val_sample`, and `test_sample` lists.

To generate new splits from the processed slide list:

```bash
python split_sample.py \
    --data_root ./hest1k_datasets \
    --dataset kidney \
    --kflod 5 \
    --seed 42
```

This writes `sample_split_flod_0.json` through `sample_split_flod_4.json`. Keep the spelling `flod` in paths and `--kflod` in commands.

The generator creates five repeated random train/validation/test splits, with approximately 80%/10%/10% of slides in each split. It does not construct disjoint test folds. If `--split_dir` is omitted from `main.py`, a single random split is generated instead.

## 🚀 Training

### Single split

This example trains HyBoxST on the supplied kidney split and then evaluates it:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root ./hest1k_datasets \
    --dataset kidney \
    --model_name hyboxst \
    --gene_list_filename selected_gene_list.txt \
    --pathway_resource_dir ./pathway_resources/kidney \
    --split_dir ./split/kidney/flod_5/sample_split_flod_0.json \
    --experiment_name kidney_HMHVG_fold0 \
    --fold 0 \
    --gpu 0 \
    --batch_size 128 \
    --epochs 100 \
    --patience 10 \
    --lr 1e-4 \
    --last_layer 11 \
    --huggingface_token "$HF_TOKEN"
```

For the base variant, use `--model_name hyboxst_base`. For HVG experiments, use `--gene_list_filename selected_hvg_gene_list.txt` and, for the full model, pathway resources matching that gene list.

### Five splits across multiple GPUs

Edit [train_data_5fold_multigpu.sh](train_data_5fold_multigpu.sh) to set `DATA_ROOT`, `DATASET`, `GENE_LIST_FILENAME`, `GENE_TAG`, `MODEL_NAME`, and `GPUS`, then run:

```bash
bash train_data_5fold_multigpu.sh
```

The current launcher uses physical GPUs `2` and `3`, runs one process per split, and waits for each batch of GPU jobs before starting the next. Each process uses logical GPU `0` through `CUDA_VISIBLE_DEVICES`. The launcher requires `HF_TOKEN`.

**Current limitation:** the final aggregation step calls `summarize_5fold_results.py`, which is absent from this tree. Training can produce individual split results, but the launcher then exits with a missing-script error. Restore the summarizer or remove the final aggregation step before using this as a complete workflow. The configured summary destination is `experiments/summary/<dataset>_<gene_tag>_<ablation_tag>_5fold/`.

### Main options

| Option | Default | Purpose |
| --- | --- | --- |
| `--model_name` | `hyboxst` | Full or base model. |
| `--epochs` | `200` | Maximum epochs; the GPU launcher sets `100`. |
| `--batch_size` | `128` | Batch size. |
| `--lr` | `0.0001` | AdamW learning rate. |
| `--patience` | `10` | Early-stopping patience based on validation loss. |
| `--seed` | `42` | Random seed. |
| `--last_layer` | `11` | Number of final UNI blocks targeted for LoRA adaptation. |
| `--pathway_box_dim` | `128` | Pathway-box dimension. |

Automatic mixed precision is enabled by default. In the current parser, passing `--amp` **disables** it (`store_false`). Run `python main.py --help` in the installed environment for the full option list.

## 🧪 Evaluation from a Checkpoint

Use `--only_test` and pass the checkpoint **directory**. Keep the model, gene list, pathway resources, split, and architecture options consistent with training:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --only_test \
    --checkpoint_path /path/to/training_run/checkpoints \
    --test_checkpoint best \
    --data_root ./hest1k_datasets \
    --dataset kidney \
    --model_name hyboxst \
    --gene_list_filename selected_gene_list.txt \
    --pathway_resource_dir ./pathway_resources/kidney \
    --split_dir ./split/kidney/flod_5/sample_split_flod_0.json \
    --experiment_name kidney_HMHVG_eval_fold0 \
    --fold 0 \
    --gpu 0 \
    --uni_weight_path ./uni_weight
```

Evaluation still loads the train/validation/test datasets. By default, it refits post-training calibration using the configured fitting split and writes calibration files into the checkpoint directory. Add `--disable_post_calibration` to skip this step. Arguments are not automatically restored from the training `config.json`.

## 📊 Outputs

With a supplied split file, a training run creates:

```text
experiments/
└── hyboxst/
    └── kidney_HMHVG_fold0/
        └── kidney/
            └── sample_split_flod_0/
                └── <timestamp>/
                    ├── config.json
                    ├── final_test_result_0.json
                    ├── samples/
                    │   ├── predict_samples.pt
                    │   └── label_samples.pt
                    └── checkpoints/
                        ├── best_model.pth
                        ├── final_model.pth
                        ├── best_ema_model.pth
                        ├── final_ema_model.pth
                        ├── gene_wise_calibrator.pth
                        └── gene_wise_calibration_stats.json
```

EMA and calibration files depend on the corresponding features being enabled. Distribution calibration additionally saves `distribution_calibration_stats.json` when fitted. Without a supplied split, the timestamp directory is directly beneath the dataset directory.

The final JSON records MSE, MAE, and mean gene-wise Pearson correlations for the top 10, 50, and 200 genes, ranked by correlation, together with evaluation settings. Prediction and label tensors have shape `(number_of_retained_test_spots, number_of_selected_genes)` and use the loader's `log2(x + 1)` expression scale. Columns follow the selected gene list; rows follow test-slide order and the loader's retained spot order.

The program prints final metrics and the result location. It does not create TensorBoard events, training log files, per-gene/per-slide diagnostic tables, or intermediate tensor dumps. If using `--experiment_result_path` to override the output location, create that directory and its `samples/` subdirectory beforehand.

## 📁 Code Layout

```text
HyBoxST/
├── model.py                       # Base/full models and LoRA setup
├── lorentz.py                     # Hyperbolic geometry operations
├── datasets/hyboxst_dataset.py     # Data loading and augmentation
├── modules/
│   ├── alignment.py               # Hierarchical alignment
│   ├── encoder.py                 # Representation encoders
│   └── pathway.py                 # Pathway decoder and box modeling
└── utils/utils.py                 # Seeding, splitting, checkpoint selection
main.py                            # Training, calibration, and evaluation
split_sample.py                    # Repeated slide-level random splits
train_data_5fold_multigpu.sh         # Multiple-GPU split launcher
pathway_resources/                 # Dataset-specific pathway masks
pretrained/                        # Bundled pretrained-model support code
```

See [LICENSE.txt](LICENSE.txt) for the repository license. Bundled pretrained components retain their upstream naming and requirements.
