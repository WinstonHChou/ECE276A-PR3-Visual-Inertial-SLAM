import numpy as np
from pr3_utils import *


if __name__ == '__main__':

  # Load the measurements
  filename = "../data/dataset00/dataset00.npy"
  v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)
  transformFromRtoLCamera = inversePose(extL_T_imu) @ extR_T_imu
  baseline = np.linalg.norm(transformFromRtoLCamera[:3,3])
  M_stereo = createStereoCalibrationMatrix(K_l, K_r, baseline)

  # (a) IMU Localization via EKF Prediction
  # ekf = ExtentedKalmanFilterInertial(initialPose=np.eye(4))
  # tau = np.diff(timestamps)
  # normalizedStamps = np.cumsum(np.concatenate(([0], tau)))   # normalized timestamp
  # poses = np.zeros((len(normalizedStamps), 3))
  # prevPose = initialPose
  # for i in len(tau):
  #   omega_t = self.omega[i-1]

  #   twist = jnp.zeros((4, 4))
  #   twist[0, 1] = -omega_t
  #   twist[1, 0] = omega_t
  #   twist[0, 3] = v_t

  #   T = T_prev @ scipy.linalg.expm(twist * dt)

  #   R = T[:3, :3]
  #   _, _, yaw = mat2euler(R, axes='sxyz')

  #   pose[0] = T[0, 3]             # x
  #   pose[1] = T[1, 3]             # y
  #   pose[2] = angle_modulus(yaw)  # theta
    
  #   T_prev = T

  # (b) Landmark Mapping via EKF Update

  # (c) Visual-Inertial SLAM

  # You may use the function below to visualize the robot pose over time
  # visualize_trajectory_2d(world_T_imu, show_ori = True)


