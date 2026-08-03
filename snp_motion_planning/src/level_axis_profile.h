#pragma once

/**
 * Make Descartes PREFER arm configurations that keep a tool axis horizontal.
 *
 * A centrifugal blast wheel throws media tangentially, so the fan spreads in the
 * plane perpendicular to the wheel's motor axis. Rolling the tool about the blast
 * axis tips that fan over and smears coverage, so the motor axis wants to stay
 * horizontal ("levelled").
 *
 * Descartes samples the roll about the tool's Z at every waypoint and picks the
 * cheapest path through the resulting ladder graph -- but the stock state
 * evaluator returns 0 for every candidate, so all rolls look equally good and the
 * choice is arbitrary. Clamping the sampling range instead is too blunt: where no
 * levelled solution exists the waypoint has NO candidates at all and the whole
 * plan fails. Scoring the states keeps every candidate available while pulling the
 * solution towards level wherever level is actually reachable.
 *
 * This is a preference, not a guarantee. Use snp_motion_planning's
 * `level_axis_weight` parameter to set how strongly it competes with the
 * smoothness cost.
 */

#include <memory>
#include <string>
#include <utility>

#include <Eigen/Geometry>
#include <descartes_light/core/state_evaluator.h>
#include <tesseract_command_language/poly/move_instruction_poly.h>
#include <tesseract_common/manipulator_info.h>
#include <tesseract_environment/environment.h>
#include <tesseract_kinematics/core/kinematic_group.h>
#include <tesseract_motion_planners/descartes/profile/descartes_default_plan_profile.h>

namespace snp_motion_planning
{
/** @brief Cost = weight * angle (rad) of `axis` (in `link`) away from horizontal. */
template <typename FloatType>
class LevelAxisStateEvaluator : public descartes_light::StateEvaluator<FloatType>
{
public:
  LevelAxisStateEvaluator(std::shared_ptr<const tesseract_kinematics::KinematicGroup> manip, std::string link,
                          Eigen::Vector3d axis, double weight)
    : manip_(std::move(manip)), link_(std::move(link)), axis_(axis.normalized()), weight_(weight)
  {
  }

  std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& solution) const override
  {
    const Eigen::VectorXd q = solution.values.template cast<double>();
    const tesseract_common::TransformMap tf = manip_->calcFwdKin(q);
    const auto it = tf.find(link_);
    if (it == tf.end())
      return std::make_pair(true, static_cast<FloatType>(0.0));

    // Component of the axis along world up; 0 => perfectly horizontal.
    const Eigen::Vector3d a = it->second.linear() * axis_;
    const double vertical = std::min(1.0, std::abs(a.z()));
    return std::make_pair(true, static_cast<FloatType>(weight_ * std::asin(vertical)));
  }

private:
  std::shared_ptr<const tesseract_kinematics::KinematicGroup> manip_;
  std::string link_;
  Eigen::Vector3d axis_;
  double weight_;
};

/** @brief DescartesDefaultPlanProfile that scores states by how level `level_axis` is. */
template <typename FloatType>
class LevelAxisDescartesPlanProfile : public tesseract_planning::DescartesDefaultPlanProfile<FloatType>
{
public:
  using Ptr = std::shared_ptr<LevelAxisDescartesPlanProfile<FloatType>>;

  std::string level_link;                        //!< link the axis belongs to (e.g. the wheel body)
  Eigen::Vector3d level_axis{ 0, 0, 1 };         //!< the axis, in that link's frame
  double level_weight{ 0.0 };                    //!< 0 disables and restores stock behaviour

  std::unique_ptr<descartes_light::StateEvaluator<FloatType>>
  createStateEvaluator(const tesseract_planning::MoveInstructionPoly& move_instruction,
                       const tesseract_common::ManipulatorInfo& composite_manip_info,
                       const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    if (level_weight <= 0.0 || level_link.empty())
      return tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createStateEvaluator(
          move_instruction, composite_manip_info, env);

    tesseract_common::ManipulatorInfo manip_info =
        composite_manip_info.getCombined(move_instruction.getManipulatorInfo());
    if (!this->manipulator_ik_solver.empty())
      manip_info.manipulator_ik_solver = this->manipulator_ik_solver;
    if (manip_info.empty())
      throw std::runtime_error("LevelAxisDescartesPlanProfile: manipulator info is empty!");

    auto manip = tesseract_planning::DescartesPlanProfile<FloatType>::createKinematicGroup(manip_info, *env);
    return std::make_unique<LevelAxisStateEvaluator<FloatType>>(manip, level_link, level_axis, level_weight);
  }

protected:
  // The task composer archives the planning problem, so any profile type it may
  // encounter has to be registered with boost::serialization -- otherwise every
  // plan dies with "unregistered class - derived class not registered".
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, const unsigned int);  // NOLINT
};

}  // namespace snp_motion_planning

BOOST_CLASS_EXPORT_KEY(snp_motion_planning::LevelAxisDescartesPlanProfile<float>)
BOOST_CLASS_EXPORT_KEY(snp_motion_planning::LevelAxisDescartesPlanProfile<double>)
