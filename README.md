# MaMoMo
MaMoMo: Masked Molecular Modeling with Graph Attention Networks

## Requirements

### 1. Dependencies

Ensure you have the following dependencies installed:

- [Python](https://www.python.org/) (>=3.6)
- [PyTorch](https://pytorch.org/) (>=1.6.0)

### 2. Python Libraries

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

- [NumPy](https://numpy.org/) (>=1.18.0)
- [Matplotlib](https://matplotlib.org/) (>=3.1.0)
- [Pytorch Lightning](https://lightning.ai/docs/pytorch/1.8.6/) (>=1.8.0,<=2.0.0)
- [DGL](https://www.dgl.ai/pages/start.html) (>=1.1.2)
- [matgl](https://pymatgen.org/installation.html) (==0.7.0)
- [Hydra](https://hydra.cc/docs/intro/)(>=1.3.1)
- [pymatgen](https://pymatgen.org/installation.html)

### 3. Hardware Requirements

This project is optimized for GPU acceleration. Ensure you have access to a compatible GPU and install the necessary CUDA toolkit and cuDNN library. Visit [PyTorch's GPU support page](https://pytorch.org/get-started/locally/) for detailed instructions.

### 4. Dataset
[//]: <> (This is also a comment.)
[//]: <> (Specify the dataset used for training and testing. Include download links or instructions for obtaining the dataset.)

[//]: <> (```bash)
[//]: <> (# Example: Download and extract dataset)
[//]: <> (wget http://example.com/dataset.zip)
[//]: <> (unzip dataset.zip -d data/)
[//]: <> (```)
### 5. Configuration

Adjust the configuration parameters inside `src/config/` files to customize the model training and evaluation settings.

### 6. Environment Setup

Create and activate a virtual environment to isolate the project dependencies.

```bash
# Example: Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 7. Running the Code

To run the training procedure, please ensure that you are in the `src` folder, then run

```bash
python train_lightning.py
```

Hydra allows configuration parameters to be overriden by command line arguments:
```bash
#Example
python train_lightning.py train.num_epochs=500 train.batch_size==32 #hydra overrides are optional and refer to parameters in src/config files
```

Data visualization and model evaluation are contained in notebooks at the moment (see `node_prediction.ipynb`, `graph_prediction.ipynb` and `node+graph_prediction.ipynb`) and is still a WIP.


