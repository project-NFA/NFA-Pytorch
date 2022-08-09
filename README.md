


## Enviroment Requirement

`pip install -r requirements.txt`


## Run a 3-layer NFA-LightGCN on Taobao dataset:


* change `ROOT_PATH` in `code/world.py`

* ` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --dataset="taobao" --topks="[20]" --recdim=64`

