echo [$(date)]: "START"
echo [$(date)]: "Creating the Python (3.8) environment"
conda create --prefix ./env python=3.8 -y
echo [$(date)]: "Activating the env"
source activate ./env
echo [$(date)]: "Installing the dev requirements"
pip install -r requirements_dev.txt
echo [$(date)]: "END"