## Requirements
- uv

## Prepare env
```sh
uv sync
```

## Get dataset
```sh
uv run scripts/1_get_original_data.py
```

## Augment dataset
```sh
uv run scripts/2_augmentation.py
```

# TODO:
- place `scripts/` in colab
