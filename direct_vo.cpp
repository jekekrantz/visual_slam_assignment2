#include <algorithm>
#include <stdio.h>
#include <thread>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <Eigen/Dense>
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <sophus/se3.hpp>

// Simple functions so that we can easily treat jets and double input similarly.
inline double value(double v) {return v;}

template<int N>
inline double value(const ceres::Jet<double, N>& v) {return v.a;}

// Templated biliner interpolation for depth values. Contains some tricks
// specific for depth images. These tricks would not be needed for
// intensity images.
template<class T>
inline T interpolate(
  const Eigen::Matrix<T, 2, 1>& pixel,
  const cv::Mat image,
  double edge_threshold = 0.1,
  double scale = 0.0002)
{
  auto clamp = [](int v, int low, int high)
  {
    return (v < low) ? low : (high < v) ? high : v;
  };

  // Clamp pixel index values to the valid range for repeating edges.
  int x0 = clamp(value(pixel(0)),     0, image.cols - 1);
  int x1 = clamp(value(pixel(0)) + 1, 0, image.cols - 1);
  int y0 = clamp(value(pixel(1)),     0, image.rows - 1);
  int y1 = clamp(value(pixel(1)) + 1, 0, image.rows - 1);

  // Extract values for the chosen pixels. Scale to get measurements in meters.
  double I00 = image.at<uint16_t>(y0, x0) * scale;
  double I01 = image.at<uint16_t>(y0, x1) * scale;
  double I10 = image.at<uint16_t>(y1, x0) * scale;
  double I11 = image.at<uint16_t>(y1, x1) * scale;

  // If there is a depth edge, interpolation does not give meaningfull values.
  // The same happens one of the pixels do not have a valid value, so
  // return 0 for invalid. It can be interestsing to remove this check to see
  // how this affects the results.
  if (std::abs(I00 - I01) > edge_threshold ||
      std::abs(I00 - I10) > edge_threshold ||
      std::abs(I00 - I11) > edge_threshold ||
      I00 == 0 || I01 == 0 || I10 == 0 || I11 == 0)
  {
    return T(0);
  }

  // Standard bilinear interpolation
  using std::floor;
  T w0 = pixel(0) - floor(pixel(0));
  T w1 = pixel(1) - floor(pixel(1));
  return (T(1) - w0) * (T(1) - w1) * T(I00) +
                 w0  * (T(1) - w1) * T(I01) +
         (T(1) - w0) *         w1  * T(I10) +
                 w0  *         w1  * T(I11);
}

// Projection functions
class PinholeCamera
{
 public:
  PinholeCamera(
    const Eigen::Vector2d& optical_center,
    const Eigen::Vector2d& focal_length)
  : optical_center_(optical_center), focal_length_(focal_length){}

  // Project a point to the image plane using a pinhole camera model.
  template<class T>
  Eigen::Matrix<T, 2, 1> project(const Eigen::Matrix<T, 3, 1>& point) const
  {
    Eigen::Matrix<T, 2, 1> pixel;
    pixel(0) = T(0); // STUDENTS: PUT THE CORRECT CODE HERE
    pixel(1) = T(0); // STUDENTS: PUT THE CORRECT CODE HERE
    return pixel;
  }

  // Generate a 3d point for a pixel and a depth value.
  template<class T>
  Eigen::Matrix<T, 3, 1> inverseProject(
      const Eigen::Matrix<T, 2, 1>& pixel,
      const T& depth) const
  {
    Eigen::Matrix<T, 3, 1> point;
    point(0) = T(0);// STUDENTS: PUT THE CORRECT CODE HERE
    point(1) = T(0);// STUDENTS: PUT THE CORRECT CODE HERE
    point(2) = T(0);// STUDENTS: PUT THE CORRECT CODE HERE
    return point;
  }

  Eigen::Vector2d optical_center_;
  Eigen::Vector2d focal_length_;
};

// Pose prior cost function
class PosePriorError
{
 public:
  PosePriorError(
    const Sophus::SE3d& frame_to_world_prior,
    const Eigen::Matrix<double,6,6>& frame_to_world_prior_information_root)
  : frame_to_world_prior_(frame_to_world_prior),
    frame_to_world_prior_information_root_(frame_to_world_prior_information_root)
  {}

  template <typename T>
  bool operator()(const T* const world_to_frame_raw, T* residuals_raw) const
  {
    const Eigen::Map<const Sophus::SE3<T>> world_to_frame(world_to_frame_raw);
    Eigen::Map<Eigen::Matrix<T,6,1>> residuals(residuals_raw);
    residuals = frame_to_world_prior_information_root_ *
                (world_to_frame * frame_to_world_prior_.cast<T>()).log();
    return true;
  }

 private:
  const Sophus::SE3d frame_to_world_prior_;
  // square root of the inverse of the covariance matrix.
  const Eigen::Matrix<double,6,6> frame_to_world_prior_information_root_;
};

// Depth error cost function
class DirectDepthError
{
 public:
  DirectDepthError(
    const PinholeCamera& camera,
    const Eigen::Vector3d& world_point,
    const cv::Mat& frame,
    const double weight)
  : camera_(camera),
    world_point_(world_point),
    frame_(frame),
    weight_(weight)
  {}
  template <typename T>
  bool operator()(const T* const world_to_frame_raw, T* residuals) const
  {
    const Eigen::Map<const Sophus::SE3<T>> world_to_frame(world_to_frame_raw);
    Eigen::Matrix<T, 3, 1> world_point = world_point_.cast<T>();
    residuals[0] = T(0); // STUDENTS: PUT THE CORRECT COST FUNCTION HERE
    return true;
  }

 private:
  const PinholeCamera& camera_;
  Eigen::Vector3d world_point_;
  const cv::Mat& frame_;
  const double weight_;
};

// Update rule for manifold updates in SE3
class SE3Plus {
 public:
  template <typename T>
  bool operator()(const T* pose_raw, const T* delta_raw, T* update_raw) const
  {
    const Eigen::Map<const Sophus::SE3<T>> pose(pose_raw);
    const Eigen::Map<const Eigen::Matrix<T, 6, 1> > delta(delta_raw);
    Eigen::Map<Sophus::SE3<T>> update(update_raw);
    update = pose * Sophus::SE3<T>::exp(delta);
    return true;
  }
};

// Very simple noise model based on the distance.
inline double noise(double distance)
{
    return 0.002 + 0.005 * distance * distance;
}

// Predict the frame_to_world transform.
// @param frame
//  Image to track.
// @param frame_to_world
//  Intial guess for the frame to world transform.
// @param keyframe
//  keyframe image to track against.
// @param keyframe_to_world
//  Keyframe to world transform.
// @param camera
//  Camera model used for projection and inverse projection.
// @param frame_to_world_prior
//  Motion model prior for the frame to world transform.
// @param frame_to_world_prior_information_root
//  The root of the inverse of the information matrix. The values given are
//  totally arbitrary.
Sophus::SE3d track(
  const cv::Mat& frame,
  const Sophus::SE3d& frame_to_world,
  const cv::Mat& keyframe,
  const Sophus::SE3d& keyframe_to_world,
  const PinholeCamera& camera,
  double relaxation,
  const Sophus::SE3d& frame_to_world_prior,
  const Eigen::Matrix<double,6,6>& frame_to_world_prior_information_root =
      1000 * Eigen::Matrix<double,6,6>::Identity())
{
  // Optimizer object
  ceres::Problem problem;

  // variable to optimize for.
  Sophus::SE3d world_to_frame = frame_to_world.inverse();


  // Add a cost function for all pixels with valud depth
  for (double u = 0; u < keyframe.cols; ++u)
  {
    for (double v = 0; v < keyframe.rows; ++v)
    {
      Eigen::Vector2d pixel(u, v);
      double d = interpolate(pixel, keyframe);

      // If the pixel has invalid depth, dont add the constraint.
      if(d == 0){continue;}

      // Standard deviation. Measurement noise + relaxation.
      // Use sum variance law to properly add the variances.
      double standard_deviation =
        std::sqrt(std::pow(relaxation, 2) + std::pow(noise(d), 2));

      // Point in world coordinates. Computed once before the cost function is
      // added to the optimizer for efficiency reasons.
      Eigen::Vector3d world_point =
        keyframe_to_world * camera.inverseProject(pixel, d);

      // STUDENTS: THIS IS WHERE YOU CHOOSE THE LOSS FUNCTION.
      // STUDENTS: TRY REPLACING THE LOSS WITH ceres::TrivialLoss AND SEE WHAT
      //           HAPPENS, ceres::TrivialLoss IS A SQUARE LOSS.
      // Read more about the available loss functions at:
      //   https://github.com/kashif/ceres-solver/blob/master/include/ceres/loss_function.h
      auto loss_function = new ceres::HuberLoss(1.5);

      // Add the cost function.
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<DirectDepthError, 1, 7>(
          new DirectDepthError(
            camera,
            world_point,
            frame,
            1.0 / standard_deviation)),
        loss_function,
        world_to_frame.data());
    }
  }

  // Add the prior cost function
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<PosePriorError, 6, 7>(
      new PosePriorError(frame_to_world_prior,
                         frame_to_world_prior_information_root)),
    new ceres::TrivialLoss(),
    world_to_frame.data());

  // Tell the optimizer to update on the se3 manifold.
  problem.AddParameterBlock(
    world_to_frame.data(), 7,
    new ceres::AutoDiffLocalParameterization<SE3Plus, 7, 6>);

  // Give settings for the solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.num_threads =
    std::max<unsigned int>(1, std::thread::hardware_concurrency());
  ceres::Solver::Summary summary;

  // Run the optimizer
  ceres::Solve(options, &problem, &summary);

  // Return the frame_to_world transform.
  return world_to_frame.inverse();
}

// Image pyramid for depth images. Contains tricks specific to depth images.
// Computes an average depth value for each pixel in the pyramid. Ignores pixels
// which lack a depth measurement.
// @param image
//  Depth image to create a pyramid for.
// @param levels
//  Number of levels in the output pyramid.
std::vector<cv::Mat> imagePyramid(
  const cv::Mat& image,
  size_t levels)
{
  cv::Mat weight(image.rows,image.cols, CV_32FC1);

  for(int i = 0; i < image.rows; ++i)
  {
    for(int j = 0; j < image.cols; ++j)
    {
      weight.at<float>(i,j) = image.at<unsigned short>(i,j) != 0;
    }
  }

  std::vector<cv::Mat> pyramid;
  pyramid.emplace_back(image);

  while(pyramid.size() < levels)
  {
    int rows = weight.rows / 2;
    int cols = weight.cols / 2;
    cv::Mat next_weight(rows, cols, CV_32FC1);
    cv::Mat next_image(rows, cols, CV_16UC1);

    for(int i = 0; i < rows; ++i)
    {
      for(int j = 0; j < cols; ++j)
      {
        float w00 = weight.at<float>(2 * i, 2 * j);
        float w01 = weight.at<float>(2 * i, 2 * j + 1);
        float w10 = weight.at<float>(2 * i + 1, 2 * j);
        float w11 = weight.at<float>(2 * i + 1, 2 * j + 1);
        float sum = w00 + w01 + w10 + w11;
        next_weight.at<float>(i,j) = sum;
        if(sum == 0)
        {
          next_image.at<uint16_t>(i,j) = 0;
        }
        else
        {
          next_image.at<uint16_t>(i,j) =
            (w00 * pyramid.back().at<uint16_t>(2 * i, 2 * j) +
             w01 * pyramid.back().at<uint16_t>(2 * i, 2 * j + 1) +
             w10 * pyramid.back().at<uint16_t>(2 * i + 1, 2 * j) +
             w11 * pyramid.back().at<uint16_t>(2 * i + 1, 2 * j + 1)) / sum;
        }
      }
    }
    weight = next_weight;
    pyramid.emplace_back(next_image);
  }
  return pyramid;
}

// Function for creating a pointcloud for the visualizer.
cv::viz::WCloud getCloud(
  const Sophus::SE3d& pose,
  const cv::Mat& image,
  const PinholeCamera& camera,
  double r,
  double g,
  double b)
{
  std::vector<Eigen::Vector3d> points;
  for (double x = 0; x < image.cols; ++x)
  {
    for (double y = 0; y < image.rows; ++y)
    {
      Eigen::Vector2d pixel(x, y);
      double d = interpolate(pixel, image);
      if(d == 0){continue;}
      points.emplace_back(pose * camera.inverseProject(pixel, d));
    }
  }

  cv::Mat points_mat(points.size(), 1, CV_64FC3);
  for (size_t j = 0; j < points.size(); ++j)
  {
    points_mat.at<cv::Vec3d>(j, 0) = cv::Vec3d(points[j](0), points[j](1), points[j](2));
  }

  return cv::viz::WCloud(points_mat, cv::viz::Color(b,g,r));
}

int main(int argc, char** argv )
{
  std::vector<PinholeCamera> camera;
  // camera parmeters for fr3
  camera.emplace_back(Eigen::Vector2d(320.1, 247.6), Eigen::Vector2d(535.4,	539.2));

  // Relaxation to allow larger outlier rejection region at coarse levels.
  std::vector<double> relaxation = {0, 0, 0, 0.03, 0.06};

  while (camera.size() < relaxation.size())
  {
      camera.emplace_back(camera.back().optical_center_ / 2, camera.back().focal_length_ / 2);
  }

  // The keyframe is a pyramid of frames.
  std::vector<cv::Mat> keyframe; 

  // The velocity is used to predict the next camera position and as a prior.
  Sophus::SE3d velocity;
  Sophus::SE3d currentframe_to_world;
  Sophus::SE3d keyframe_to_world;

  // Visualization
  cv::viz::Viz3d viewer("Viz Demo");
  viewer.setBackgroundColor();

  // Lets run at 10 hz to make run faster and make development easier.
  for (int i = 1; i < argc; i += 3)
  {
    std::vector<cv::Mat> currentframe =
      imagePyramid(cv::imread(argv[i], -1), camera.size());

    if (!keyframe.empty())
    {
      Sophus::SE3d previousframe_to_world = currentframe_to_world;
      currentframe_to_world = velocity * currentframe_to_world; 
      Sophus::SE3d currentframe_to_world_prior = currentframe_to_world;

      // STUDENTS: Try using only c = 0 and see what happens. This is equivalent
      //           to not using an image pyramid. The for loop does not run the
      //           two most highresolution layers for performance reasons.
      for(int c = camera.size() - 1; c >= 2; --c)
      {
        currentframe_to_world = track(currentframe[c], currentframe_to_world,
                                      keyframe[c], keyframe_to_world, camera[c],
                                      relaxation[c], currentframe_to_world_prior);
      }
      velocity = currentframe_to_world * previousframe_to_world.inverse();
    }

    // STUDENTS: Figure out some reasonable values here. This is intended for
    //           you to learn about  how keyframing affects the results
    //           (replace the -1 values).
    Sophus::SE3d distance_to_keyframe = currentframe_to_world *
                                        keyframe_to_world.inverse();
    double translation_distance = distance_to_keyframe.translation().norm();
    double rotation_distance = distance_to_keyframe.so3().log().norm();
    if (keyframe.empty() || translation_distance > -1 || rotation_distance > -1)
    {
      keyframe = currentframe;
      keyframe_to_world = currentframe_to_world;
    }

    // Visualize the estimate.
    viewer.showWidget("coordinate_system" + std::to_string(i),
                      cv::viz::WCameraPosition(0.05),
                      cv::Affine3d(Eigen::Affine3d(currentframe_to_world.
                                                   matrix())));
    viewer.showWidget("currentframe",
                      getCloud(currentframe_to_world, currentframe.front(),
                               camera.front(), 0, 255, 0));
    viewer.showWidget("keyframe",
                      getCloud(keyframe_to_world, keyframe.front(),
                               camera.front(), 255, 0, 0));
    viewer.spinOnce();
  }

  viewer.spin();
  return 0;
}
