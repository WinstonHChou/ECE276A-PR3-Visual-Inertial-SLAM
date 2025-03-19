# ECE276A-PR3-Visual-Inertial-SLAM
WI 25 ECE 276A Project 3: Visual-Inertial SLAM

## Course Overview
This is Project 3 for [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, taught by Professor [Nikolay Atanasov](https://natanaso.github.io/).

## Project Description
The project involves ..:

## Prerequisites
The code is only tested with miniconda environment in **WSL2** to ensure that `open3d` can draw 3D pictures. If you're using Linux machine or others, **Please make sure your `open3d` works as it should.**
- Install [WSL2](https://dev.to/brayandiazc/install-wsl-from-the-microsoft-store-111h) and [External X Server (if needed)](https://www.google.com/search?q=VcXsrv)
- Miniconda Installed: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ```
    After installing, close and reopen your terminal application or refresh it by running the following command:
    ```bash
    source ~/miniconda3/bin/activate
    conda init --all
    ```
- Use `conda` to create a `python3.10` virtual environment (`ece276a_pr3`), and install required packages:
    ```bash
    conda create -n ece276a_pr3 python=3.10
    conda install -c conda-forge libstdcxx-ng -n ece276a_pr3
    conda activate ece276a_pr3
    pip3 install -r requirements.txt
    ```
- Whenever creating a **new terminal session**, do:
    ```bash
    conda deactivate
    conda activate ece276a_pr3
    export XDG_SESSION_TYPE=x11
    ```
- If you **cannot open any 3D model through `open3d`**, and any error happens as follows, 
    ```python
    [Open3D WARNING] GLFW Error: Wayland: The platform does not support setting the window position
    [Open3D WARNING] Failed to initialize GLEW.
    [Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.
    ```
    **Check the following GitHub Issues: [Open3D Github Issue #6872](https://github.com/isl-org/Open3D/issues/6872), [Open3D Github Issue #5126](https://github.com/isl-org/Open3D/issues/5126)**

## Dataset Setup
1. Download the dataset from [OneDrive]()
2. Organize the folder structure as follows:
    ```text
    .
    └── data
      ├── dataset00
      │ ├── dataset00.npy
      │ ├── dataset00_imgs.npy
      │ ├── dataset00_l.mp4
      │ └── dataset00_r.mp4
      ├── dataset00
      │ ├── dataset01.npy
      │ ├── dataset01_imgs.npy
      │ ├── dataset01_l.mp4
      │ └── dataset01_r.mp4
      ├── dataset02
      │ ├── dataset02.npy
      │ ├── dataset02_imgs.npy
      │ ├── dataset02_l.mp4
      │ └── dataset02_r.mp4
      └── README.md
    ```

## Running the Project
### Standard Run:
- __`dataset=1`__: Select dataset number (1-11)
```python
python3 src/main.py dataset=1
```
### Optional Arguments:
- __`iterations=1`__: Select desired iterations (Default: 300)
- __`stepSize=1`__: Select desired step size (Default: 0.025)
- __`rough=true`__: Choose to roughly plot the panorama

__Example:__
```python
python3 src/main.py dataset=1 iterations=150 stepSize=0.05 rough=true
```

## Features
- Datasets are labelled 1-11 (10, 11 are from `data/testset/`)
- Generates **cylindrical projection panorama images** for datasets 1, 2, 8, 9, 10, 11

## Output Graphs
### Euler Angles Comparison
1. Predicted quaternions from Motion Model with IMU's angular velocities as input
2. Gradient descent processed quaternions
3. VICON data as Ground Truth

### Linear Acceleration Comparison
1. Linear acceleration from predicted quaternions using Observation Model
2. Linear acceleration from Gradient descent processed quaternions using Observation Model
3. IMU data (bias corrected)

## Sample Results (Dataset 2)
### Euler Angles Comparison
![image](assets/2_AngVel.jpg)

### Linear Acceleration Comparison
![image](assets/2_Accel.jpg)

### Panorama
![image](assets/panorama_2.jpg)

## Technical Details
- Quaternions and rotation matrices converted using `transforms3d`
- Plots done by using `matplotlib`
- Gradient descent used for IMU data refinement

## Acknowledgments
This project is part of the **ECE 276A** course at **UC San Diego**, which is inspired by the teaching and research of various professors in the field of robotics and estimation