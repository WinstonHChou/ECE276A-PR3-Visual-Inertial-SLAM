import numpy as np
from pr3_utils import *


if __name__ == '__main__':

  # Load the measurements
  dataset = 0
  filename = f"../data/dataset0{dataset}/dataset0{dataset}.npy"
  v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)
  transformFromRtoLCamera = extL_T_imu @ inversePose(extR_T_imu)
  baseline = np.linalg.norm(transformFromRtoLCamera[:3,3])
  M_stereo = createStereoCalibrationMatrix(K_l, K_r, baseline)

  downSampleInterval = 8 # USER INPUT
  featuresDownSampled = features[:,0:-1:downSampleInterval,:]
  numOfLandmarks = int(featuresDownSampled.shape[1]) # Number of Landmarks: M
  seenTracker = np.zeros(numOfLandmarks, dtype=bool)
  print(features.shape[0])

  tau = np.diff(timestamps)
  normalizedStamps = np.cumsum(np.concatenate(([0], tau)))   # normalized timestamp
  intialPoseMean = np.eye(4)
  intialPoseCovariance = 0.1 * np.eye(6) # USER INPUT
  motionModelNoise = float(0.1) # USER INPUT
  motionModelCovariance = motionModelNoise * np.eye(6)

  landmarksMean = np.zeros((3*numOfLandmarks,1))
  landmarksCovariancePriorNoise = float(0.1) # USER INPUT
  landmarksCovariance = landmarksCovariancePriorNoise * np.eye(3*numOfLandmarks)
  observationModelNoise = float(0.1) # USER INPUT

  # %% (a) IMU Localization via EKF Prediction
  ekf = ExtentedKalmanFilterInertial(M_stereo, inversePose(extL_T_imu), initialPose=intialPoseMean)
  
  poses = np.zeros((len(normalizedStamps), 4, 4))
  poses[0,:,:] = intialPoseMean
  poseCovariance = intialPoseCovariance
  for i in tqdm(range(len(tau))):
    
    # EKF Prediction
    poses[i+1,:,:], poseCovariance = ekf.ekfInertialPredict(v_t[i,:], w_t[i,:], tau[i], poses[i,:,:], poseCovariance, motionModelCovariance)

  fig, ax = visualize_trajectory_2d(poses, path_name="EKF Predicted", show_ori = True)
  plt.show()

  # %% (b) Landmark Mapping via EKF Update
  for i in tqdm(range(len(tau))):
    newObservations = featuresDownSampled[:,:,i].transpose().flatten()
    newObservationsBool = featuresValid(newObservations)
    observationsForFirstTime = firstTimeObserved(seenTracker, newObservationsBool)
    newObservationsToInitializeMeans = newObservations.reshape(numOfLandmarks, 4)[observationsForFirstTime,:]
    worldFrameNewObservations = []

    # Initialize Landmarks if observed for the first time
    for i in range(newObservationsToInitializeMeans.shape[0]):
      cameraFramePoint = getCameraFramePointFromPixelObservation(newObservationsToInitializeMeans[i], M_stereo)
      worldFramePoint = poses[i,:,:] @ inversePose(extL_T_imu) @ cameraFramePoint
      worldFrameNewObservations.append(worldFramePoint[:3])

    newMeansIndexTracker = 0

    for i in range(numOfLandmarks):
      if observationsForFirstTime[i]:
        landmarksMean[i*3:i*3+3,:] = worldFrameNewObservations[newMeansIndexTracker]
        landmarksCovariance[i,i] = landmarksCovariancePriorNoise
        newMeansIndexTracker = newMeansIndexTracker + 1

    seenTracker = seenTracker | observationsForFirstTime

    # EKF Update for Landmark Only
    # landmarksMean, landmarksCovariance = ekf.ekfLandmarkUpdate(v_t[i,:], w_t[i,:], tau[i], landmarksMean, landmarksCovariance, observationModelNoise, , newObservations)

  # fig, ax = visualize_trajectory_2d(poses, path_name="EKF Predicted", show_ori = True)
  # plt.show()

  # %% (c) Visual-Inertial SLAM

  # You may use the function below to visualize the robot pose over time
  # visualize_trajectory_2d(world_T_imu, show_ori = True)
  # plt.show()


