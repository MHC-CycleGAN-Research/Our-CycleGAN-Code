### Here is the dataset we use:
https://drive.google.com/drive/folders/1nO7h8bEnPhuSS-EzQ7g2zBK6NIajmIpE?usp=sharing

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
