import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler, euler2mat
import scipy
from tqdm import tqdm

def load_data(file_name: str):
    '''
    function to read visual features, IMU measurements, and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npy"
    Output:
        linear_velocity: velocity measurements in IMU frame
            with shape t*3
        angular_velocity: angular velocity measurements in IMU frame
            with shape t*3
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        t: time stamp
            with shape t*1
        K_l: leftcamera intrinsic matrix
            with shape 3*3
        K_r: right camera intrinsic matrix
            with shape 3*3
        extL_T_imu: extrinsic transformation from imu frame to left camera, in SE(3).
            with shape 4*4
        extL_T_imu: extrinsic transformation from imu frame to right camera, in SE(3).
            with shape 4*4
    '''
    data = np.load(file_name, allow_pickle = True).item()
    v_t = data["v_t"] # linear velocities
    w_t = data["w_t"] # angular velocities
    timestamps = data["timestamps"] # UNIX timestamps
    features = data["features"] # 4 x num_features x t : pixel coordinates of the visual features
    K_l = data["K_l"] # intrinsic calibration matrix of left camera
    K_r = data["K_r"] # intrinsic calibration matrix of right camera
    extL_T_imu = data["extL_T_imu"] # transformation from imu frame to left camera frame
    extR_T_imu = data["extR_T_imu"] # transformation from imu frame to right camera frame
    
    return v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(10, 8))
    n_pose = pose.shape[0]
    ax.plot(pose[:,0,3],pose[:,1,3],'r-',label=path_name)
    ax.scatter(pose[0,0,3],pose[0,1,3],marker='s',label="start")
    ax.scatter(pose[-1,0,3],pose[-1,1,3],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[i,:3,:3])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[select_ori_index,0,3],pose[select_ori_index,1,3],dx,dy,\
            color="b",units="xy",width=1, headlength=0.002,headaxislength=0.001)
            
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()

    return fig, ax




def projection(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  r = n x 4 = ph/ph[...,2] = normalized z axis coordinates
  '''  
  return ph/ph[...,2,None]
  
def projectionJacobian(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  J = n x 4 x 4 = Jacobian of ph/ph[...,2]
  '''  
  J = np.zeros(ph.shape+(4,))
  iph2 = 1.0/ph[...,2]
  ph2ph2 = ph[...,2]**2
  J[...,0,0], J[...,1,1],J[...,3,3] = iph2,iph2,iph2
  J[...,0,2] = -ph[...,0]/ph2ph2
  J[...,1,2] = -ph[...,1]/ph2ph2
  J[...,3,2] = -ph[...,3]/ph2ph2
  return J


def inversePose(T):
  '''
  @Input:
    T = n x 4 x 4 = n elements of SE(3)
  @Output:
    iT = n x 4 x 4 = inverse of T
  '''
  iT = np.empty_like(T)
  iT[...,0,0], iT[...,0,1], iT[...,0,2] = T[...,0,0], T[...,1,0], T[...,2,0] 
  iT[...,1,0], iT[...,1,1], iT[...,1,2] = T[...,0,1], T[...,1,1], T[...,2,1] 
  iT[...,2,0], iT[...,2,1], iT[...,2,2] = T[...,0,2], T[...,1,2], T[...,2,2]
  iT[...,:3,3] = -np.squeeze(iT[...,:3,:3] @ T[...,:3,3,None])
  iT[...,3,:] = T[...,3,:]
  return iT


def axangle2skew(a):
  '''
  converts an n x 3 axis-angle to an n x 3 x 3 skew symmetric matrix 
  '''
  S = np.empty(a.shape[:-1]+(3,3))
  S[...,0,0].fill(0)
  S[...,0,1] =-a[...,2]
  S[...,0,2] = a[...,1]
  S[...,1,0] = a[...,2]
  S[...,1,1].fill(0)
  S[...,1,2] =-a[...,0]
  S[...,2,0] =-a[...,1]
  S[...,2,1] = a[...,0]
  S[...,2,2].fill(0)
  return S

def axangle2twist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of se(3)
  '''
  T = np.zeros(x.shape[:-1]+(4,4))
  T[...,0,1] =-x[...,5]
  T[...,0,2] = x[...,4]
  T[...,0,3] = x[...,0]
  T[...,1,0] = x[...,5]
  T[...,1,2] =-x[...,3]
  T[...,1,3] = x[...,1]
  T[...,2,0] =-x[...,4]
  T[...,2,1] = x[...,3]
  T[...,2,3] = x[...,2]
  return T

def twist2axangle(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 6 axis-angle 
  '''
  return T[...,[0,1,2,2,0,1],[3,3,3,1,2,0]]

def axangle2adtwist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    A = n x 6 x 6 = n elements of ad(se(3))
  '''
  A = np.zeros(x.shape+(6,))
  A[...,0,1] =-x[...,5]
  A[...,0,2] = x[...,4]
  A[...,0,4] =-x[...,2]
  A[...,0,5] = x[...,1]
  
  A[...,1,0] = x[...,5]
  A[...,1,2] =-x[...,3]
  A[...,1,3] = x[...,2]
  A[...,1,5] =-x[...,0]
  
  A[...,2,0] =-x[...,4]
  A[...,2,1] = x[...,3]
  A[...,2,3] =-x[...,1]
  A[...,2,4] = x[...,0]
  
  A[...,3,4] =-x[...,5] 
  A[...,3,5] = x[...,4] 
  A[...,4,3] = x[...,5]
  A[...,4,5] =-x[...,3]   
  A[...,5,3] =-x[...,4]
  A[...,5,4] = x[...,3]
  return A

def twist2pose(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 4 x 4 pose (SE3) matrix 
  '''
  rotang = np.sqrt(np.sum(T[...,[2,0,1],[1,2,0]]**2,axis=-1)[...,None,None]) # n x 1
  Tn = np.nan_to_num(T / rotang)
  Tn2 = Tn@Tn
  Tn3 = Tn@Tn2
  eye = np.zeros_like(T)
  eye[...,[0,1,2,3],[0,1,2,3]] = 1.0
  return eye + T + (1.0 - np.cos(rotang))*Tn2 + (rotang - np.sin(rotang))*Tn3
  
def axangle2pose(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of SE(3)
  '''
  return twist2pose(axangle2twist(x))


def pose2adpose(T):
  '''
  converts an n x 4 x 4 pose (SE3) matrix to an n x 6 x 6 adjoint pose (ad(SE3)) matrix 
  '''
  calT = np.empty(T.shape[:-2]+(6,6))
  calT[...,:3,:3] = T[...,:3,:3]
  calT[...,:3,3:] = axangle2skew(T[...,:3,3]) @ T[...,:3,:3]
  calT[...,3:,:3] = np.zeros(T.shape[:-2]+(3,3))
  calT[...,3:,3:] = T[...,:3,:3]
  return calT

def hatmap(x):
  """
  compute hat map
  hatmap(x): R3 -> so(3)
  """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  return np.array([[0,-x3,x2],[x3,0,-x1],[-x2,x1,0]])

def hatmapR6(q):
  v = q[:3][:]
  w = q[3:][:]
  return createTwistMatrix(v.flatten(),w.flatten())

def createTwistMatrix(v,w):
  twist = np.eye(4)
  wHat = hatmap(w)
  twist[:3,:3] = wHat
  twist[:3,3] = v
  twist[3,3] = 0
  return twist

def createAdjointTwist(v,w):
  adjoint = np.zeros((6,6))
  wHat = hatmap(w)
  vHat = hatmap(v)
  adjoint[:3,:3] = wHat
  adjoint[3:,3:] = wHat
  adjoint[:3,3:] = vHat
  return adjoint
  
def createPose(R, p):
  p = np.atleast_2d(p).transpose()
  result = np.hstack((R, p))
  result = np.vstack((result, np.array([0,0,0,1])))
  return result

### Stereo Camera
transformCameraToOptical = np.array([[0,-1, 0,0],
                                     [0, 0,-1,0],
                                     [1, 0, 0,0],
                                     [0, 0, 0,1]])

def createStereoCalibrationMatrix(K_left, K_right, b):
  fsu_l, fsv_l, cu_l, cv_l = K_left[0,0], K_left[1,1], K_left[0,2], K_left[1,2]
  fsu_r, fsv_r, cu_r, cv_r = K_right[0,0], K_right[1,1], K_right[0,2], K_right[1,2]
  M_stereo = np.array([[fsu_l,     0, cu_l,        0],
                       [    0, fsv_l, cv_l,        0],
                       [fsu_r,     0, cu_r, -fsu_r*b],
                       [    0, fsv_r, cv_r,        0]])
  return M_stereo

def getCameraFramePointFromPixelObservation(feature, K_l, K_r, transformFromRtoLCamera):
  fsu_l = K_l[0,0]
  fsv_l = K_l[1,1]
  fsu_r = K_r[0,0]
  fsv_r = K_r[1,1]
  cu_l = K_l[0, 2]
  cv_l = K_l[1, 2]
  cu_r = K_r[0, 2]
  cv_r = K_r[1, 2]
  R = transformFromRtoLCamera[:3,:3]
  p = transformFromRtoLCamera[:3,3]
  b = np.linalg.norm(p)

  z_l = feature[:2]
  z_r = feature[2:]
  normalizedDisparity = (z_l[0] - cu_l)/fsu_l - (z_r[0] - cu_r)/fsu_r
  if abs(normalizedDisparity) < 5E-3:
    return None

  m_z = b / normalizedDisparity
  m_x = (z_l[0] - cu_l)/fsu_l * m_z
  m_y = (z_l[1] - cv_l)/fsv_l * m_z

  m = np.vstack((m_x, m_y, m_z, 1))
  m_CameraFrame = inversePose(transformCameraToOptical) @ m
  return m_CameraFrame

def firstTimeObserved(tracker, newObservationsBool):
  """
  returns True if landmark is observed for the first time
  """
  trackerInt = tracker.astype(int)
  newObservationsBoolInt = newObservationsBool.astype(int)
  firstTimeObserved = trackerInt - newObservationsBoolInt
  return firstTimeObserved == -1

def featuresValid(features):
  """
  Determines if a landmark has been observed by checking if all its feature values are valid.
    Input:
        features (4 x N array): Matrix with 4 rows corresponding to u_L, v_L, u_R, v_R per feature.
    Returns:
        valid (1D array): Boolean array of size N indicating if each landmark has valid observations.
  """
  numOfLandmarks_t = int(features.shape[0] / 4)
  featuresReshaped = features.reshape(numOfLandmarks_t,4)
  featuresReshaped = featuresReshaped[:,0]
  landmarkObserved = featuresReshaped != -1
  return landmarkObserved

def convertNormalLandmarksToHomogenous(q):
  numOfLandmarks_t = int(q.shape[0] / 3)
  landmarks = q.reshape(numOfLandmarks_t,3)
  landmarks = np.column_stack((landmarks, np.repeat(1, np.shape(landmarks)[0])))
  landmarks = landmarks.reshape(numOfLandmarks_t * 4, 1)
  return landmarks

def convertHomogeneousLandmarksToNormal(q):
  numOfLandmarks_t = int(q.shape[0] / 4)
  landmarks = q.reshape(numOfLandmarks_t,4)
  landmarks = landmarks[:,:3]
  landmarks = landmarks.reshape(numOfLandmarks_t * 3, 1)
  return landmarks

def piProjection(q):
  """
  q is R3
  """
  q3 = q[2][0]
  return (1/q3) * q

def piProjectionLandmarksHomogeneous(q):
  """
  q is homogeneous coordinates
  """
  numOfLandmarks_t = int(q.shape[0] / 4)
  landmarks = q.reshape(numOfLandmarks_t,4)
  q3 = landmarks[:,2]
  landmarks = landmarks.transpose()
  result = np.atleast_2d((1 / q3)) * landmarks
  result = result.transpose()
  result = result.reshape(numOfLandmarks_t * 4, 1)
  return result

def observationModel(ImuToWorldPose, ImuToCameraFramePose, landmarkWorldCoords, stereoCalibMatrix, landmarkObservedInNewObservations):
  numOfLandmarks_t = int(landmarkWorldCoords.shape[0] / 3)
  landmarkWorldCoordsMasked = landmarkWorldCoords.reshape(numOfLandmarks_t, 3)
  landmarkWorldCoordsMasked = landmarkWorldCoordsMasked[landmarkObservedInNewObservations, :]
  landmarkWorldCoordsMasked = landmarkWorldCoordsMasked.reshape(-1,1)
  numOfLandmarks_t = int(landmarkWorldCoordsMasked.shape[0] / 3)

  mu = convertNormalLandmarksToHomogenous(landmarkWorldCoordsMasked)
  T = scipy.linalg.block_diag(*([ImuToCameraFramePose @ np.linalg.pinv(ImuToWorldPose)]*numOfLandmarks_t))
  K = scipy.linalg.block_diag(*([stereoCalibMatrix]*numOfLandmarks_t))
  result = K @ piProjectionLandmarksHomogeneous(T @ mu)
  return result

def observationModelJacobianRespectToLandmarks(ImuToWorldPose, ImuToCameraFramePose, landmarkCoordsMeansPrior, stereoCalibMatrix, landmarkObservedInNewObservations):
  newObservationsCount = np.count_nonzero(landmarkObservedInNewObservations == True)
  numOfLandmarks_t = int(landmarkCoordsMeansPrior.shape[0] / 3)
  H = np.zeros((4*newObservationsCount, 3*numOfLandmarks_t))
  HRowTracker = 0
  for i in range(numOfLandmarks_t):
    if landmarkObservedInNewObservations[i]:
      landmarkCoordMeanPrior = landmarkCoordsMeansPrior[3*i:3*i+3,:]
      landmarkCoordMeanPrior = np.vstack((landmarkCoordMeanPrior,1))
      H_i_i = stereoCameraJacobian(ImuToWorldPose, ImuToCameraFramePose, landmarkCoordMeanPrior, stereoCalibMatrix)
      H[4*HRowTracker:4*HRowTracker+4,3*i:3*i+3] = H_i_i
      HRowTracker = HRowTracker + 1
  return H

def observationModelJacobianRespectToPose(ImuToWorldPoseMeanPrior, ImuToCameraFramePose, landmarkCoords, stereoCalibMatrix, landmarkObservedInNewObservations):
  newObservationsCount = np.count_nonzero(landmarkObservedInNewObservations == True)
  numOfLandmarks_t = int(landmarkCoords.shape[0] / 3)
  inverseImuToWorldPoseMeanPrior = np.linalg.pinv(ImuToWorldPoseMeanPrior)
  H = np.zeros((4*newObservationsCount, 6))
  HRowTracker = 0
  for i in range(numOfLandmarks_t):
    if landmarkObservedInNewObservations[i]:
      landmarkCoord = landmarkCoords[3*i:3*i+3,:]
      landmarkCoord = np.vstack((landmarkCoord,1))
      H_t1_i = -stereoCalibMatrix @ derivativeProjectionFunction(ImuToCameraFramePose @ inverseImuToWorldPoseMeanPrior @ landmarkCoord) @ ImuToCameraFramePose @ odotMap(inverseImuToWorldPoseMeanPrior @ landmarkCoord)
      H[4*HRowTracker:4*HRowTracker+4,:] = H_t1_i
      HRowTracker = HRowTracker + 1
  return H

def odotMap(q):
  s = q[:3,:]
  result = np.zeros((4,6))
  result[:3,:3] = np.eye(3)
  result[:3,3:] = -hatmap(s.flatten())
  return result

def derivativeProjectionFunction(q):
  """
  q is homogeneous coordinates
  """
  q1 = q[0][0]
  q2 = q[1][0]
  q3 = q[2][0]
  q4 = q[3][0]
  result = (1 / q3) * np.array([[1,0,-q1/q3,0],
                                [0,1,-q2/q3,0],
                                [0,0,0,0],
                                [0,0,-q4/q3,1]])
  return result

def stereoCameraJacobian(ImuToWorldPose, ImuToCameraFramePose, landmarkWorldCoord, stereoCalibMatrix):
  P = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0]])
  worldToImuPose = np.linalg.pinv(ImuToWorldPose)
  d_proj = derivativeProjectionFunction(transformCameraToOptical @ ImuToCameraFramePose @ worldToImuPose @ landmarkWorldCoord)
  result = stereoCalibMatrix @ d_proj @ transformCameraToOptical @ ImuToCameraFramePose @ worldToImuPose @ P.transpose()
  return result

class ExtentedKalmanFilterInertial:
  def __init__(self, stereoCalibrationMatrix, ImuToCameraFramePose, initialPose = np.eye(4)):
    self.initialPose = initialPose
    self.stereoCalibrationMatrix = stereoCalibrationMatrix
    self.ImuToCameraFramePose = ImuToCameraFramePose

  def ekfInertialPredict(self, v, w, tau, priorMean, priorCovariance, motionModelNoiseCovariance):
    twist = createTwistMatrix(v, w)
    adjoint = createAdjointTwist(v, w)
    mu = priorMean @ scipy.linalg.expm(tau * twist)
    F = scipy.linalg.expm(-tau * adjoint)
    sigma = (F @ priorCovariance @ F.transpose()) + tau * motionModelNoiseCovariance
    return mu, sigma, F

  def ekfLandmarkUpdate(self, ImuToWorldPose, priorMeans, priorCovariance, observationModelNoise, newObservations):
    
    if (np.count_nonzero(featuresValid(newObservations) == True) == 0):
      return priorMeans, priorCovariance
    landmarkObservedInNewObservations = featuresValid(newObservations)
    landmarksObservedRange = np.argwhere(landmarkObservedInNewObservations == True)
    firstLandmarkObservedIndex = int(landmarksObservedRange[0])
    numOfLandmarks_t = int(newObservations.shape[0] / 4)
    newObservationsMasked = newObservations.reshape(numOfLandmarks_t, 4)
    newObservationsMasked = newObservationsMasked[landmarkObservedInNewObservations, :]
    newObservationsMasked = newObservationsMasked.reshape(-1,1)
    numOfLandmarks_t = int(newObservationsMasked.shape[0] / 4)

    # Define relevant data to update
    relevantCovariance = priorCovariance[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3, firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3]

    H = observationModelJacobianRespectToLandmarks(ImuToWorldPose, self.ImuToCameraFramePose, priorMeans, self.stereoCalibrationMatrix, landmarkObservedInNewObservations)
    H = H[:,firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3]
    observationModelCovariance = observationModelNoise * np.eye(4 * numOfLandmarks_t)
    
    # The diagonals that correspond to the newly observed landmarks will change, everything else remains the same to speed up the process
    observationModelCovariance = observationModelNoise * np.eye(4 * numOfLandmarks_t)
    K =  relevantCovariance @ H.transpose() @ np.linalg.pinv(H @ relevantCovariance @ H.transpose() + observationModelCovariance)
    mu = priorMeans[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] + K @ (newObservationsMasked - observationModel(ImuToWorldPose, self.ImuToCameraFramePose, priorMeans, self.stereoCalibrationMatrix, landmarkObservedInNewObservations))
    sigma = K @ H
    sigma = (np.eye(sigma.shape[0]) - K @ H) @ relevantCovariance

    full_mu = priorMeans.copy()
    full_mu[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] = mu
    full_sigma = priorCovariance.copy()
    full_sigma[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3, firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] = sigma

    return full_mu, full_sigma
  
  def ekfPredict(self, v, w, tau, priorPoseMean, priorCovariance, priorLandmarkMean, motionModelNoiseCovariance):
    dimension = priorCovariance.shape[0]
    landmarkCount = int((dimension - 6)/3)
    poseCovariance = priorCovariance[landmarkCount * 3:, landmarkCount * 3:]
    bottomLeftCovariance = priorCovariance[landmarkCount * 3:, :landmarkCount * 3]
    topRightCovariance = priorCovariance[:landmarkCount * 3, landmarkCount * 3:]
    poseMean, poseCovariance, F = self.ekfInertialPredict(v, w, tau, priorPoseMean, poseCovariance, motionModelNoiseCovariance)

    # Update all covaraince
    priorCovariance[landmarkCount * 3:, landmarkCount * 3:] = poseCovariance
    priorCovariance[landmarkCount * 3:, :landmarkCount * 3] = F @ bottomLeftCovariance
    priorCovariance[:landmarkCount * 3, landmarkCount * 3:] = topRightCovariance @ F.transpose()
    return priorCovariance, priorLandmarkMean, poseMean

  
  def ekfUpdate(self, landmarkMeanPrior, poseMeanPrior, totalCovariancePrior, observationModelNoise, newObservations):
    dimension = totalCovariancePrior.shape[0]
    landmarkCount = int((dimension - 6)/3)
    landmarkCovariancePrior = totalCovariancePrior[:landmarkCount * 3, :landmarkCount * 3]
    poseCovariancePrior = totalCovariancePrior[landmarkCount * 3:, landmarkCount * 3:]

    landmarkObservedInNewObservations = featuresValid(newObservations)

    # Compute Combined Jacobian
    HRespectToLandmark = observationModelJacobianRespectToLandmarks(poseMeanPrior, self.ImuToCameraFramePose, landmarkMeanPrior, self.stereoCalibrationMatrix, landmarkObservedInNewObservations)
    HRespectToPose = observationModelJacobianRespectToPose(poseMeanPrior, self.ImuToCameraFramePose, landmarkMeanPrior, self.stereoCalibrationMatrix, landmarkObservedInNewObservations)
    HSLAM = np.hstack((HRespectToLandmark, HRespectToPose))

    numberOfLandmarks = int(newObservations.shape[0] / 4)
    newObservationsMasked = newObservations.reshape(numberOfLandmarks, 4)
    newObservationsMasked = newObservationsMasked[landmarkObservedInNewObservations, :]
    newObservationsMasked = newObservationsMasked.reshape(-1,1)
    numberOfLandmarks = int(newObservationsMasked.shape[0] / 4)

    if numberOfLandmarks == 0:
      return totalCovariancePrior, landmarkMeanPrior, poseMeanPrior

    observationModelCovariance = observationModelNoise * np.eye(4 * numberOfLandmarks)

    # Compute the Combined Kalman Gain
    KSLAM = totalCovariancePrior @ HSLAM.transpose() @ np.linalg.pinv(HSLAM @ totalCovariancePrior @ HSLAM.transpose() + observationModelCovariance)
    KLandmark = KSLAM[:3 * landmarkCount, :]
    KPose = KSLAM[3 * landmarkCount:, :]

    observationModelPrediction = observationModel(poseMeanPrior, self.ImuToCameraFramePose, landmarkMeanPrior, self.stereoCalibrationMatrix, landmarkObservedInNewObservations)

    poseMean, poseCovariance = self.ekfSLAMInertialUpdate(poseMeanPrior, poseCovariancePrior, newObservations, KPose, observationModelPrediction, HRespectToPose)

    landmarkMean, landmarkCovariance = self.ekfSLAMLandmarkUpdate(landmarkMeanPrior, landmarkCovariancePrior, newObservations, KLandmark, observationModelPrediction, HRespectToLandmark)

    totalCovariancePrior[:landmarkCount * 3, :landmarkCount * 3] = landmarkCovariance
    totalCovariancePrior[landmarkCount * 3:, landmarkCount * 3:] = poseCovariance

    return totalCovariancePrior, landmarkMean, poseMean
  
  def ekfSLAMInertialUpdate(self, ImuToWorldPoseMeanPrior, ImuToWorldPoseCovariancePrior, newObservations, K, observationModelPrediction, H):
    landmarkObservedInNewObservations = featuresValid(newObservations)

    if (np.count_nonzero(landmarkObservedInNewObservations == True) == 0):
      return ImuToWorldPoseMeanPrior, ImuToWorldPoseCovariancePrior
    
    numberOfLandmarks = int(newObservations.shape[0] / 4)
    newObservationsMasked = newObservations.reshape(numberOfLandmarks, 4)
    newObservationsMasked = newObservationsMasked[landmarkObservedInNewObservations, :]
    newObservationsMasked = newObservationsMasked.reshape(-1,1)
    numberOfLandmarks = int(newObservationsMasked.shape[0] / 4)

    mu = ImuToWorldPoseMeanPrior @ scipy.linalg.expm(hatmapR6(K @ (newObservationsMasked - observationModelPrediction)))
    sigma = K @ H
    sigma = (np.eye(sigma.shape[0]) - sigma) @ ImuToWorldPoseCovariancePrior
    return mu, sigma

  def ekfSLAMLandmarkUpdate(self, priorMeans, priorCovariance, newObservations, KLandmark, observationModelPrediction, HRespectToLandmark):
    
    if (np.count_nonzero(featuresValid(newObservations) == True) == 0):
      return priorMeans, priorCovariance
    landmarkObservedInNewObservations = featuresValid(newObservations)
    landmarksObservedRange = np.argwhere(landmarkObservedInNewObservations == True)
    firstLandmarkObservedIndex = int(landmarksObservedRange[0])
    numOfLandmarks_t = int(newObservations.shape[0] / 4)
    newObservationsMasked = newObservations.reshape(numOfLandmarks_t, 4)
    newObservationsMasked = newObservationsMasked[landmarkObservedInNewObservations, :]
    newObservationsMasked = newObservationsMasked.reshape(-1,1)
    numOfLandmarks_t = int(newObservationsMasked.shape[0] / 4)

    # Extract Kalman Gain and Define relevant data to update
    K = KLandmark[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3,:]
    relevantCovariance = priorCovariance[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3, firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3]

    H = HRespectToLandmark[:,firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3]
    mu = priorMeans[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] + K @ (newObservationsMasked - observationModelPrediction)
    sigma = K @ H
    sigma = (np.eye(sigma.shape[0]) - K @ H) @ relevantCovariance

    full_mu = priorMeans.copy()
    full_mu[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] = mu
    full_sigma = priorCovariance.copy()
    full_sigma[firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3, firstLandmarkObservedIndex*3:(firstLandmarkObservedIndex + numOfLandmarks_t)*3] = sigma

    return full_mu, full_sigma
