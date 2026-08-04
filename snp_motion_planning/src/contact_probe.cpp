/**
 * @file contact_probe.cpp
 * @brief Ask Tesseract directly what is in contact at a given joint state.
 *
 * Why this exists: with collision checking OFF the planner returns a properly
 * levelled path (0.21 deg mean tilt); with it ON, planning fails and Descartes
 * reports ~97/104 failed vertices. Offline analysis of the same joint states says
 * every body is clear -- nearest is 9.3 mm to wrist2_link, which the SRDF disables,
 * then 214 mm to the plate and 400 mm to the upper arm. Something disagrees.
 *
 * Rather than keep guessing, this loads the SAME environment the planning server
 * builds, drops the robot into a state we measured as clear, and runs Tesseract's
 * own discrete contact manager. Two outcomes, both decisive:
 *   - contacts reported -> Tesseract names the pair and depth, and the offline
 *     model is wrong somewhere specific.
 *   - nothing reported  -> the vertices are being rejected for a reason that is
 *     not state collision.
 *
 * Usage (joint values in radians, j1..j6):
 *   ros2 run snp_motion_planning contact_probe --ros-args \
 *     -p robot_description:="$(xacro ...)" -p robot_description_semantic:="$(cat ...)" \
 *     -p joints:="[0.1328, -0.8599, 2.33, -4.3384, -1.6987, -0.7553]" \
 *     -p margin:=0.0
 */
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tesseract_environment/environment.h>
#include <tesseract_rosutils/utils.h>
#include <tesseract_collision/core/discrete_contact_manager.h>
#include <tesseract_collision/core/types.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("contact_probe");

  node->declare_parameter<std::string>("robot_description", "");
  node->declare_parameter<std::string>("robot_description_semantic", "");
  node->declare_parameter<std::vector<double>>("joints", {});
  node->declare_parameter<double>("margin", 0.0);
  node->declare_parameter<std::vector<std::string>>(
      "joint_names", { "j1", "j2", "j3", "j4", "j5", "j6" });

  const auto urdf = node->get_parameter("robot_description").as_string();
  const auto srdf = node->get_parameter("robot_description_semantic").as_string();
  const auto joints = node->get_parameter("joints").as_double_array();
  const auto names = node->get_parameter("joint_names").as_string_array();
  const auto margin = node->get_parameter("margin").as_double();

  if (urdf.empty() || srdf.empty() || joints.size() != names.size())
  {
    RCLCPP_ERROR(node->get_logger(), "need robot_description, robot_description_semantic, "
                                     "and one joint value per joint name");
    return 1;
  }

  auto env = std::make_shared<tesseract_environment::Environment>();
  auto locator = std::make_shared<tesseract_rosutils::ROSResourceLocator>();
  if (!env->init(urdf, srdf, locator))
  {
    RCLCPP_ERROR(node->get_logger(), "environment init failed");
    return 1;
  }

  Eigen::VectorXd q(static_cast<Eigen::Index>(joints.size()));
  for (std::size_t i = 0; i < joints.size(); ++i)
    q[static_cast<Eigen::Index>(i)] = joints[i];
  env->setState(names, q);

  auto mgr = env->getDiscreteContactManager();
  if (mgr == nullptr)
  {
    RCLCPP_ERROR(node->get_logger(), "no discrete contact manager");
    return 1;
  }
  // Same margin the Descartes profile is given via min_contact_distance.
  mgr->setDefaultCollisionMarginData(margin);
  mgr->setActiveCollisionObjects(env->getActiveLinkNames());
  mgr->setCollisionObjectsTransform(env->getState().link_transforms);

  std::cout << "\n=== contact probe ===\n";
  std::cout << "links in env      : " << env->getLinkNames().size() << "\n";
  std::cout << "active links      : " << env->getActiveLinkNames().size() << "\n";
  std::cout << "margin            : " << margin << " m\n";
  std::cout << "state             : ";
  for (std::size_t i = 0; i < joints.size(); ++i)
    std::cout << names[i] << "=" << joints[i] << (i + 1 < joints.size() ? ", " : "\n");

  // CLOSEST reports the nearest pair even when nothing overlaps, which tells us
  // the true separation Tesseract believes in -- not just pass/fail.
  tesseract_collision::ContactResultMap results;
  tesseract_collision::ContactRequest req(tesseract_collision::ContactTestType::ALL);
  mgr->contactTest(results, req);

  std::cout << "contact pairs     : " << results.size() << "\n";
  if (results.empty())
  {
    std::cout << "\nNO CONTACTS -- Tesseract agrees this state is clear.\n"
              << "So the Descartes vertices are NOT being rejected by state collision.\n";
  }
  else
  {
    std::cout << "\n";
    for (const auto& pair : results)
    {
      for (const auto& r : pair.second)
      {
        std::cout << "  " << r.link_names[0] << "  <->  " << r.link_names[1]
                  << "   distance " << r.distance << " m\n";
      }
    }
    std::cout << "\nTesseract DOES see contact here; the offline model is wrong.\n";
  }

  rclcpp::shutdown();
  return 0;
}
