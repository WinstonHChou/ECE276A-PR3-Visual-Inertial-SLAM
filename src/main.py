import numpy as np
from pr3_utils import *


if __name__ == '__main__':

  # Load the measurements
  dataset = 0
  filename = f"../data/dataset0{dataset}/dataset0{dataset}.npy"
  v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)
  transformFromRtoLCamera = inversePose(extL_T_imu) @ extR_T_imu
  baseline = np.linalg.norm(transformFromRtoLCamera[:3,3])
  M_stereo = createStereoCalibrationMatrix(K_l, K_r, baseline)

  downSampleInterval = 8 # USER INPUT
  featuresDownSampled = features[:,0:-1:downSampleInterval,:]
  numOfLandmarks = int(featuresDownSampled.shape[1]) # Number of Landmarks: M

  tau = np.diff(timestamps)
  normalizedStamps = np.cumsum(np.concatenate(([0], tau)))   # normalized timestamp
  intialPoseMean = np.eye(4)
  intialPoseCovariance = 0.1 * np.eye(6) # USER INPUT
  motionModelNoise = float(0.0) # USER INPUT
  motionModelCovariance = motionModelNoise * np.eye(6)

  landmarksCovariance = 0.01 * np.eye(3*numOfLandmarks) # USER INPUT

  # %% (a) IMU Localization via EKF Prediction
  ekf = ExtentedKalmanFilterInertial(M_stereo, initialPose=intialPoseMean)
  
  poses = np.zeros((len(normalizedStamps), 4, 4))
  poses[0,:,:] = intialPoseMean
  poseCovariance = intialPoseCovariance
  for i in tqdm(range(len(tau))):
    
    # EKF Prediction
    poses[i+1,:,:], poseCovariance = ekf.ekfPredict(v_t[i,:], w_t[i,:], tau[i], poses[i,:,:], poseCovariance, motionModelCovariance)

  fig, ax = visualize_trajectory_2d(poses, path_name="EKF Predicted", show_ori = True)
  plt.show()

  # %% (b) Landmark Mapping via EKF Update

  # %% (c) Visual-Inertial SLAM

  # You may use the function below to visualize the robot pose over time
  # visualize_trajectory_2d(world_T_imu, show_ori = True)
  # plt.show()


