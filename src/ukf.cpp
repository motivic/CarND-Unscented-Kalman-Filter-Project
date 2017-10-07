#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Set flag to indicate the state has not been initialized
  is_initialized_ = false;

  // Set timestamp to 0
  time_us_ = 0;

  // Initialize the process covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Set the state dimension
  n_x_ = 5;

  // Set the augmented state dimension
  n_aug_ = 7;

  // Set the sigma point spreading parameter
  lambda_ = 3-n_aug_;

  // Initialize the sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Set the weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_/(lambda_+n_aug_);
  weights_.tail(2*n_aug_) = VectorXd::Ones(2*n_aug_)*0.5/(lambda_+n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Initialize with radar measurement
      float rho = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);

      float px = rho*cos(theta);
      float py = rho*sin(theta);

      x_ << px, py, 0, theta, 0;
      time_us_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize with lidar measurement
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
      time_us_ = meas_package.timestamp_;
    }
    is_initialized_ = true;
    return;
  }
  
  /**
   * Prediction
   */
  float dt = (meas_package.timestamp_-time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  /**
   * Update
   */
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar update
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Lidar update
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Create augmented state vector
  VectorXd x_aug = VectorXd(n_aug_);
  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
 
  // Create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_-n_x_, n_aug_-n_x_) << std_a_*std_a_, 0,
                                                       0, std_yawdd_*std_yawdd_;
  // Calculate square root of the covariance matrix 
  MatrixXd A = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = x_aug.replicate(1, n_aug_)+sqrt(lambda_+n_aug_)*A;
  Xsig_aug.block(0, 1+n_aug_, n_aug_, n_aug_) = x_aug.replicate(1, n_aug_)-sqrt(lambda_+n_aug_)*A;

  // Predict sigma points
  double dt_2 = delta_t*delta_t;
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // Predict state values
    double px_p, py_p;
    
    if (fabs(yawd) > 0.001) {
      px_p = p_x+v/yawd*(sin(yaw+yawd*delta_t)-sin(yaw));
      py_p = p_y+v/yawd*(cos(yaw)-cos(yaw+yawd*delta_t));
    }
    else {
      px_p = p_x+v*delta_t*cos(yaw);
      py_p = p_y+v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw+yawd*delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p += 0.5*nu_a*dt_2*cos(yaw);
    py_p += 0.5*nu_a*dt_2*sin(yaw);
    v_p += nu_a*delta_t;

    yaw_p += 0.5*nu_yawdd*dt_2;
    yawd_p += nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // Predict mean and covariance
  x_.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    x_ += weights_(i)*Xsig_pred_.col(i);
  }
  

  P_.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    // Angle normalization
    while (x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i)*x_diff*x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  // Transform sigma points into measurement space
  Zsig = Xsig_pred_.topLeftCorner(n_z, 2*n_aug_+1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  // Measurement covariance matrix and cross correlation matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    VectorXd z_diff = Zsig.col(i)-z_pred;
    S += weights_(i)*z_diff*z_diff.transpose();

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    Tc += weights_(i)*x_diff*z_diff.transpose();
  }

  // Add measurement noise to the covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S += R;
  
  // Kalman gain
  MatrixXd K = Tc*S.inverse();
  // Residual
  VectorXd z_diff = meas_package.raw_measurements_-z_pred;

  // Update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();
 
  // Calculate Lidar NIS
  NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
  
  // Transform sigma points into measurement space
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // Measurement model
    Zsig(0,i) = sqrt(p_x*p_x+p_y*p_y);
    Zsig(1,i) = atan2(p_y, p_x);
    Zsig(2,i) = (p_x*v1+p_y*v2)/Zsig(0,i);
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    z_pred += weights_(i)*Zsig.col(i);
  }

  // Measurement covariance matrix and cross correlation matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (unsigned int i=0; i<2*n_aug_+1; ++i) {
    VectorXd z_diff = Zsig.col(i)-z_pred;
    // Angle normalization
    while (z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i)*z_diff*z_diff.transpose();

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    // Angle normalization
    while (x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc += weights_(i)*x_diff*z_diff.transpose();
  }

  // Add measurement noise to the covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S += R;
  
  // Kalman gain
  MatrixXd K = Tc*S.inverse();
  // Residual
  VectorXd z_diff = meas_package.raw_measurements_-z_pred;
  // Angle normalization
  while (z_diff(1)>M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // Update state mean and covariance matrix
  x_ += K*z_diff;
  P_ -= K*S*K.transpose();

  // Calculate Radar NIS
  NIS_radar_ = z_diff.transpose()*S.inverse()*z_diff;
}