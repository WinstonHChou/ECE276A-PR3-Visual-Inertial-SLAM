# ECE276A-PR3-Visual-Inertial-SLAM
WI 25 ECE 276A Project 3: Visual-Inertial SLAM

## Course Overview
This is Project 3 for [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, taught by Professor [Nikolay Atanasov](https://natanaso.github.io/).

## Project Description
This project focuses on implementing a visual-inertial simultaneous localization and mapping (SLAM) system using an extended Kalman filter (EKF).
- The system integrates measurements from an inertial measurement unit (IMU) and a stereo camera.
- The SLAM process involves two main steps: an EKF prediction step based on IMU kinematics to estimate the robot's pose and an EKF update step using visual observations to refine landmark positions.
- The project assumes known extrinsic and intrinsic calibration parameters for the sensors. 

## Prerequisites
The code is only tested with miniconda environment in **WSL2**.
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
    ```

## Dataset Setup
1. Download the dataset from [OneDrive](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/w3chou_ucsd_edu/EUBt6V9A56lAsncXhD-9_BIBe6Bsq7GR8HBnDoLPDEbaMA?e=dmgq6g)
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
- __`dataset=0`__: Select dataset number (0-2)
```python
python3 src/main.py dataset=0
```

## Features
- Datasets are labelled 0-2
- Generates **Visual-Inertial SLAM Trajectory**

## Output Graphs

### Visual-Inertial SLAM Trajectory
1. Linear acceleration from predicted quaternions using Observation Model
2. Linear acceleration from Gradient descent processed quaternions using Observation Model
3. IMU data (bias corrected)

## Acknowledgments
This project is part of the **ECE 276A** course at **UC San Diego**, which is inspired by the teaching and research of various professors in the field of robotics and estimation