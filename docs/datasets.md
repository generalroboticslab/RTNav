# Prepare Datasets

Set the directory to download datasets.

```bash
export DATA_DIR=/path/to/data
```

## Preparation Environment

Create a lightweight environment for setup:

```bash
conda create -n rtnav python==3.9 -y && conda activate rtnav
conda install huggingface_hub gdown habitat-sim -c conda-forge -c aihabitat
```

## HM3D Scenes

Download the HM3D validation scenes. You will need a [Matterport account](https://matterport.com/partners/meta):

```bash
export MATTERPORT_TOKEN_ID=your_matterport_token_id
export MATTERPORT_TOKEN_SECRET=your_matterport_token_secret
python -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 --data-path "$DATA_DIR"
```

## HM3D ObjectNav Episodes

Download both ObjectNav episode versions:

```bash
mkdir -p "$DATA_DIR/datasets/objectnav/hm3d"
for version in 1 2; do
  wget -nc -P "$DATA_DIR" \
    "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v${version}/objectnav_hm3d_v${version}.zip"
  unzip "$DATA_DIR/objectnav_hm3d_v${version}.zip" -d "$DATA_DIR"
  mv "$DATA_DIR/objectnav_hm3d_v${version}" \
    "$DATA_DIR/datasets/objectnav/hm3d/v${version}"
  rm "$DATA_DIR/objectnav_hm3d_v${version}.zip"
done

mkdir -p "$DATA_DIR/scene_datasets"
ln -s ../versioned_data/hm3d-0.2/hm3d \
  "$DATA_DIR/scene_datasets/hm3d_v0.2"
```

## OVON Episodes

```bash
hf download nyokoyama/hm3d_ovon \
  hm3d.tar.gz text_embeddings/siglip.pkl \
  --repo-type dataset --local-dir "$DATA_DIR/hm3d_ovon"
mkdir -p "$DATA_DIR/datasets/ovon/hm3d"
tar -xzf "$DATA_DIR/hm3d_ovon/hm3d.tar.gz" -C "$DATA_DIR/datasets/ovon/"
find "$DATA_DIR/datasets/ovon" -type f -name '._*' -delete # remove mac
```
