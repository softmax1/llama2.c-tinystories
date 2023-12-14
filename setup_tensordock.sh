conda activate tensorml

git config --global credential.helper store
git config --global user.name $GIT_USERNAME
git config --global user.email $GIT_EMAIL
sudo apt install git-lfs
git lfs install

git clone https://github.com/softmax1/llama2.c-tinystories
git checkout tensordock

echo 'alias skyy="cd ~/llama2.c-tinystories && conda activate tensorml"' >> ~/.bashrc
source ~/.bashrc

skyy
pip install -r requirements.txt

python tinystories.py download
python tinystories.py pretokenize
python login.py