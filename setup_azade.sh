conda env create -f environment.yaml
source ~/.bashrc
conda activate ldm
pip install torchmetrics==0.6.0
pip install opencv-contrib-python-headless==4.2.0.32
pip install torch==1.13.1 torchvision==0.14.1
pip install pytorch-lightning==1.4.2
pip install omegaconf kornia
pip install packaging==21.3
pip install SceneGraphParser
python -m spacy download en