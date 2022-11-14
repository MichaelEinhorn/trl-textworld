pip install --upgrade pip

conda install -c anaconda protobuf==3.18.1
pip install gym==0.23.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge torchinfo
conda install -c conda-forge transformers pytorch-lightning 
pip install deepspeed
pip install textworld
conda install -c fastai accelerate
pip install numexpr -U

git config --global user.email "einhorn.michael1@gmail.com"
git config --global user.name "MichaelEinhorn"
