#pragma once

/**
 * The two things this cell asks of Descartes that stock Descartes does not do.
 *
 * ONE: prefer arm configurations that keep a tool axis horizontal.
 * TWO: refuse to change arm configuration between two adjacent waypoints.
 *
 * Both live on one profile class because Descartes takes exactly one plan
 * profile per move instruction, so there is one slot to put them in.
 *
 * --- ONE ---------------------------------------------------------------
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
 *
 * --- TWO ---------------------------------------------------------------
 *
 * Descartes scores an edge by joint distance and nothing else. Nothing caps how
 * far a joint may travel between two adjacent states, and edge collision
 * checking is off (`enable_edge_collision = false`) because leaving it on is
 * far too slow. So where two arm configurations both reach a raster line, the
 * cheapest path through the ladder graph is free to step from one to the other
 * between two states 65 mm apart. Both states are collision-free, which is all
 * Descartes ever checks -- the sweep between them is not.
 *
 * TrajOpt, which gets that seed, DOES check between states (LVS_DISCRETE). It
 * then has to remove a 60 deg wrist flip while holding the tool on the line,
 * cannot, and the whole raster fails -- reported to the operator as "out of
 * reach or collides", about a path every waypoint of which is reachable.
 *
 * Measured 2026-08-28 on the sofa back-wall scan, 10 passes, 267 states: 95 %
 * of state-to-state transitions moved every joint less than 10 deg, and 12
 * moved one joint 20-63 deg. All 12 sat in two clusters, in the middle of two
 * blasting passes, and those same states are where TrajOpt's collision and
 * Cartesian constraints were violated from its very first iteration. Probing
 * the same two passes at 15 mm showed a smooth chain existed throughout, with
 * 2-3 IK branches per pose -- so the flip was never necessary, only cheap.
 *
 * `max_joint_step` is the cap, in radians, applied per joint per edge. 0 leaves
 * Descartes exactly as it was.
 */

#include <memory>
#include <string>
#include <utility>

#include <Eigen/Geometry>
#include <descartes_light/core/edge_evaluator.h>
#include <descartes_light/core/state_evaluator.h>
#include <descartes_light/edge_evaluators/compound_edge_evaluator.h>
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

/** @brief Rejects an edge that asks any ONE joint to move more than `limit` radians.
 *
 * Cost is left at zero: the Euclidean evaluator it is compounded with already
 * prices joint travel, and this one only has an opinion about what is allowed.
 */
template <typename FloatType>
class MaxJointStepEdgeEvaluator : public descartes_light::EdgeEvaluator<FloatType>
{
public:
  explicit MaxJointStepEdgeEvaluator(double limit) : limit_(static_cast<FloatType>(limit)) {}

  std::pair<bool, FloatType> evaluate(const descartes_light::State<FloatType>& start,
                                      const descartes_light::State<FloatType>& end) const override
  {
    const auto step = (end.values - start.values).cwiseAbs().maxCoeff();
    return std::make_pair(step <= limit_, static_cast<FloatType>(0.0));
  }

private:
  FloatType limit_;
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
  double max_joint_step{ 0.0 };                  //!< rad, per joint per edge; 0 disables

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

  std::unique_ptr<descartes_light::EdgeEvaluator<FloatType>>
  createEdgeEvaluator(const tesseract_planning::MoveInstructionPoly& move_instruction,
                      const tesseract_common::ManipulatorInfo& composite_manip_info,
                      const std::shared_ptr<const tesseract_environment::Environment>& env) const override
  {
    auto base = tesseract_planning::DescartesDefaultPlanProfile<FloatType>::createEdgeEvaluator(
        move_instruction, composite_manip_info, env);
    if (max_joint_step <= 0.0)
      return base;

    // Compound ANDs validity and sums cost, so the stock evaluator keeps pricing
    // the edge and this one only removes the ones that jump.
    auto compound = std::make_unique<descartes_light::CompoundEdgeEvaluator<FloatType>>();
    compound->evaluators.push_back(std::shared_ptr<descartes_light::EdgeEvaluator<FloatType>>(std::move(base)));
    compound->evaluators.push_back(std::make_shared<MaxJointStepEdgeEvaluator<FloatType>>(max_joint_step));
    return compound;
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
