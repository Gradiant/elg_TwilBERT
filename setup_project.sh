# Build the environment (conda)

conda create -n twilbertservs python=3.6 --yes

# activate the environment (conda)
eval "$(conda shell.bash hook)"
conda activate twilbertservs

# Cloning the original repository

git clone https://github.com/jogonba2/TWilBert


# Install requirements
pip install -r requirements.txt


# Export pythonpath
bash pythonpath.sh

# copy the config used in the tests
mkdir -p TWilBert/configs/microservs/
cp config/config_large_server.json TWilBert/configs/microservs/
cp config/config_single_hateeval19_base.json TWilBert/configs/microservs/
cp config/config_labelling_single_hateeval19_base.json TWilBert/configs/microservs/
cp config/config_labelling_single_hateeval19_large.json TWilBert/configs/microservs/


# Overwrite the single labelling file
cp scripts/single_labeling.py TWilBert/twilbert/applications/bert/
cp scripts/single_labeling_api.py TWilBert/
cp scripts/serve.py TWilBert/
