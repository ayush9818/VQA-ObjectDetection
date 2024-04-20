# VQA-ObjectDetection



## Installation

1. **Clone the repository**

   If you're using git for version control, clone the project repository to your local machine using the following command:

   ```bash
   git clone https://github.com/ayush9818/VQA-ObjectDetection.git
   cd VQA-ObjectDetection
   ```
2. **Create the Conda environment**

   ```bash
   conda create -n <env_name> python=3.9
   ```
3. **Activate Environment**

    ```bash
    conda activate <env_name>
    ```
4. **Install Dependencies**

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install chardet
    pip install -r requirements.txt
    ```
    
## Experiments

| Model  | Epochs| Batch Size| Lr  | Optimizer|Freeze Layers| Train Accuracy| Test Accuracy|
| -------|-------|-----------|-----|----------|-------------|---------------|--------------|
| VILT   | 50    | 128       | 5e-5| Adam     |     0       |98.14%          | 33.83%     |
| VILT   | 40    | 256       | 2e-3| Adam     |     5       |         |     |