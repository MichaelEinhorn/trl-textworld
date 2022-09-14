cd /home/ubuntu/SHARDONNAY/trl
git pull
cp -r /home/ubuntu/SHARDONNAY/trl /home/ubuntu/trl
pip install --upgrade pip

pip install protobuf==3.18.1
pip install gym==0.23.1
pip install torch torchvision torchaudio -U --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchinfo
pip install transformers
pip install pytorch-lightning deepspeed
pip install textworld
pip install accelerate

echo "copying transformer cache"
mkdir -p ~/.cache/huggingface/transformers && cp -rv ~/SHARDONNAY/transformers ~/.cache/huggingface

git config --global user.email "einhorn.michael1@gmail.com"
git config --global user.name "MichaelEinhorn"
