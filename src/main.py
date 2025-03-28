import numpy as np
from pr3_utils import *
import sys

def parse_args(argv):
    parsed_args = {}
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key == 'dataset':
                parsed_args['dataset'] = int(value)
    
    # Set default values if not provided
    parsed_args.setdefault('dataset', None)
    
    return parsed_args


if __name__ == '__main__':

  # Load the measurements
  args = parse_args(sys.argv)

  try:
      dataset = int(args['dataset'])
  except:
      raise ValueError('Please select a dataset to process. Use: "dataset=<0-2>"')

  filename = f"data/dataset0{dataset}/dataset0{dataset}.npy"
  v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)
  K_l_inv = np.linalg.pinv(K_l)
  K_r_inv = np.linalg.pinv(K_r)
  transformFromRtoLCamera = extL_T_imu @ inversePose(extR_T_imu)
  baseline = np.linalg.norm(transformFromRtoLCamera[:3,3])
  M_stereo = createStereoCalibrationMatrix(K_l, K_r, baseline)

  downSampleInterval = 25                                                       # USER INPUT
  featuresDownSampled = features[:,0:-1:downSampleInterval,:]
  numOfLandmarks = int(featuresDownSampled.shape[1]) # Number of Landmarks: M
  seenTracker = np.zeros(numOfLandmarks, dtype=bool)

  tau = np.diff(timestamps)
  normalizedStamps = np.cumsum(np.concatenate(([0], tau)))   # normalized timestamp
  intialPoseMean = np.eye(4)
  intialPoseCovariance = 0.001 * np.eye(6)                                        # USER INPUT
  motionModelNoise = float(0.001)                                                  # USER INPUT
  motionModelCovariance = motionModelNoise * np.eye(6)

  landmarksMean = np.zeros((3*numOfLandmarks,1))
  landmarksCovariancePriorNoise = float(0.01)                                     # USER INPUT
  landmarksCovariance = landmarksCovariancePriorNoise * np.eye(3*numOfLandmarks)
  observationModelNoise = float(0.05)                                            # USER INPUT

  # %% (a) IMU Localization via EKF Prediction
  ekf = ExtentedKalmanFilterInertial(M_stereo, inversePose(extL_T_imu), initialPose=intialPoseMean)
  
  inertialPoses = np.zeros((len(normalizedStamps), 4, 4))
  inertialPoses[0,:,:] = intialPoseMean
  poseCovariance = intialPoseCovariance
  for i in tqdm(range(len(tau))):
    
    # EKF Prediction
    inertialPoses[i+1,:,:], poseCovariance, _ = ekf.ekfInertialPredict(v_t[i,:], w_t[i,:], tau[i], inertialPoses[i,:,:], poseCovariance, motionModelCovariance)

  fig, ax = visualize_trajectory_2d(inertialPoses, path_name="EKF Predicted", show_ori = False)
  ax.set_xlim(min(inertialPoses[:,0,3]) - 10, max(inertialPoses[:,0,3]) + 10)
  ax.set_ylim(min(inertialPoses[:,1,3]) - 10, max(inertialPoses[:,1,3]) + 10)
  # plt.show()

  # %% (b) Landmark Mapping via EKF Update
  landmarksMeanPrior = np.zeros((3*numOfLandmarks,1))
  for i in tqdm(range(len(tau))):
    newObservations = featuresDownSampled[:,:,i].transpose().flatten()
    newObservationsBool = featuresValid(newObservations)
    observationsForFirstTime = firstTimeObserved(seenTracker, newObservationsBool)
    newObservationsToInitializeMeans = newObservations.reshape(numOfLandmarks, 4)[observationsForFirstTime,:]
    worldFrameNewObservations = []

    # Initialize Landmarks if observed for the first time
    for ii in range(newObservationsToInitializeMeans.shape[0]):
      cameraFramePoint = getCameraFramePointFromPixelObservation(newObservationsToInitializeMeans[ii], K_l, K_r, transformFromRtoLCamera)
      if cameraFramePoint is not None:
        worldFramePoint = inertialPoses[i,:,:] @ inversePose(extL_T_imu) @ cameraFramePoint
        worldFrameNewObservations.append(worldFramePoint[:3])

    newMeansIndexTracker = 0

    for ii in range(numOfLandmarks):
      if observationsForFirstTime[ii]:
        # Check bounds before accessing the list
        if newMeansIndexTracker < len(worldFrameNewObservations):
          landmarksMeanPrior[ii*3:ii*3+3,:] = worldFrameNewObservations[newMeansIndexTracker] # just save for plotting
          landmarksMean[ii*3:ii*3+3,:] = worldFrameNewObservations[newMeansIndexTracker]
          landmarksCovariance[ii,ii] = landmarksCovariancePriorNoise
          newMeansIndexTracker += 1

    seenTracker = seenTracker | observationsForFirstTime

    # EKF Update for Landmark Only
    landmarksMean, landmarksCovariance = ekf.ekfLandmarkUpdate(inertialPoses[i,:,:], landmarksMean, landmarksCovariance, observationModelNoise, newObservations)

  landmarksPriorReshaped = landmarksMeanPrior.reshape(int(landmarksMeanPrior.shape[0] / 3), 3)
  landmarksReshaped = landmarksMean.reshape(int(landmarksMean.shape[0] / 3), 3)
  ax.scatter(landmarksPriorReshaped[:,0],landmarksPriorReshaped[:,1],color='blue',s=4)
  ax.scatter(landmarksReshaped[:,0],landmarksReshaped[:,1],color='lime',s=4)
  # plt.show()

  # %% (c) Visual-Inertial SLAM

  ekfSLAM = ExtentedKalmanFilterInertial(M_stereo, inversePose(extL_T_imu), initialPose=intialPoseMean)

  slamPoses = np.zeros((len(normalizedStamps), 4, 4))
  slamPoses[0,:,:] = intialPoseMean
  slamLandmarksMeanPrior = np.zeros((3*numOfLandmarks,1))
  slamLandmarksMean = np.zeros((3*numOfLandmarks,1))

  allCovariance = np.eye(numOfLandmarks*3 + 6)
  slamLandmarksCovariance = landmarksCovariancePriorNoise * np.eye(3*numOfLandmarks)
  allCovariance[:numOfLandmarks*3, :numOfLandmarks*3] = slamLandmarksCovariance
  allCovariance[numOfLandmarks*3:, numOfLandmarks*3:] = intialPoseCovariance
  seenTracker = np.zeros(numOfLandmarks, dtype=bool)
  for i in tqdm(range(len(tau))):
    
    # EKF Prediction
    allCovariance, slamLandmarksMean, slamPoses[i+1,:,:] = ekfSLAM.ekfPredict(v_t[i,:], w_t[i,:], tau[i], slamPoses[i,:,:], allCovariance, slamLandmarksMean, motionModelCovariance)

    # Intialize Observation
    newObservations = featuresDownSampled[:,:,i].transpose().flatten()
    newObservationsBool = featuresValid(newObservations)
    observationsForFirstTime = firstTimeObserved(seenTracker, newObservationsBool)
    newObservationsToInitializeMeans = newObservations.reshape(numOfLandmarks, 4)[observationsForFirstTime,:]
    worldFrameNewObservations = []

    # Initialize Landmarks if observed for the first time
    for ii in range(newObservationsToInitializeMeans.shape[0]):
      cameraFramePoint = getCameraFramePointFromPixelObservation(newObservationsToInitializeMeans[ii], K_l, K_r, transformFromRtoLCamera)
      if cameraFramePoint is not None:
        worldFramePoint = slamPoses[i,:,:] @ inversePose(extL_T_imu) @ cameraFramePoint
        worldFrameNewObservations.append(worldFramePoint[:3])

    newMeansIndexTracker = 0

    for ii in range(numOfLandmarks):
      if observationsForFirstTime[ii]:
        # Check bounds before accessing the list
        if newMeansIndexTracker < len(worldFrameNewObservations):
          slamLandmarksMeanPrior[ii*3:ii*3+3,:] = worldFrameNewObservations[newMeansIndexTracker] # just save for plotting
          slamLandmarksMean[ii*3:ii*3+3,:] = worldFrameNewObservations[newMeansIndexTracker]
          slamLandmarksCovariance[ii,ii] = landmarksCovariancePriorNoise
          newMeansIndexTracker += 1

    seenTracker = seenTracker | observationsForFirstTime

    # EKF Update for Landmark Only
    allCovariance, slamLandmarksMean, slamPoses[i+1,:,:] = ekfSLAM.ekfUpdate(slamLandmarksMean, slamPoses[i+1,:,:], allCovariance, observationModelNoise, newObservations)

  # You may use the function below to visualize the robot pose over time
  fig, ax = visualize_trajectory_2d(slamPoses, path_name="EKF SLAM", show_ori = True)
  ax.set_xlim(min(slamPoses[:,0,3]) - 10, max(slamPoses[:,0,3]) + 10)
  ax.set_ylim(min(slamPoses[:,1,3]) - 10, max(slamPoses[:,1,3]) + 10)

  slamLandmarksPriorReshaped = slamLandmarksMeanPrior.reshape(int(slamLandmarksMeanPrior.shape[0] / 3), 3)
  slamLandmarksReshaped = slamLandmarksMean.reshape(int(slamLandmarksMean.shape[0] / 3), 3)
  ax.scatter(slamLandmarksPriorReshaped[:,0],slamLandmarksPriorReshaped[:,1],color='blue',s=4)
  ax.scatter(slamLandmarksReshaped[:,0],slamLandmarksReshaped[:,1],color='lime',s=4)
  plt.show()


