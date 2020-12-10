### Here is the dataset we use:
https://digital.lib.washington.edu/researchworks/handle/1773/45396

### System setup:
`~/anaconda3/envs/tf-gpu/bin/pip install imageio`
`~/anaconda3/envs/tf-gpu/bin/pip install scipy`
`~/anaconda3/envs/tf-gpu/bin/pip install click`
`cd path-to-code/FinalProject`
`conda activate tf-gpu`


### Command to generate data:
`python create_cyclegan_dataset.py`

### Command to train the model:
`python -m main --to_train=1`

### Command to test the model:
`python -m main --to_train=0 --checkpoint_dir=/path-to-checkpoint-file`

### Optional: Command to load segmentation masks from source location:
`python copy_files_from_source.py`
