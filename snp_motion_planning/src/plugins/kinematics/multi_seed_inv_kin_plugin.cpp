/**
 * Multi-seed inverse kinematics.
 *
 * Tesseract's Descartes wrapper calls the IK solver ONCE per candidate pose with
 * a single seed. KDL's Levenberg-Marquardt solver is a local method, so from one
 * seed it converges to at most one solution -- and frequently to none, or to one
 * that violates the joint limits and is then discarded. Measured on the FR20
 * blasting cell: sweeping the tool roll at waypoints across a plate, LMA from a
 * single seed found solutions at only ~3/9 waypoints, while the same solver from
 * three seeds found 9/9. The arm could reach every one of those poses; the
 * planner simply never found them, which made otherwise-valid tool paths (e.g.
 * ones holding the blast fan level) fail with "LadderGraphSolver failed to build
 * graph".
 *
 * This wraps any chain solver and retries it from several seeds -- the caller's
 * own seed, a configured list of named postures, and optionally some random ones
 * drawn inside the joint limits -- returning every distinct in-limit solution.
 * That is what the graph search needs: alternatives to choose between.
 */

#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <Eigen/Geometry>
#include <boost/core/demangle.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <tesseract_kinematics/core/inverse_kinematics.h>
#include <tesseract_kinematics/core/kinematics_plugin_factory.h>
#include <tesseract_kinematics/kdl/kdl_inv_kin_chain_lma.h>
#include <tesseract_scene_graph/graph.h>
#include <tesseract_scene_graph/joint.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

namespace snp_motion_planning
{
/** @brief Retries an inner IK solver from several seeds and returns all distinct solutions. */
class MultiSeedInvKin : public tesseract_kinematics::InverseKinematics
{
public:
  MultiSeedInvKin(std::shared_ptr<tesseract_kinematics::InverseKinematics> inner,
                  std::vector<Eigen::VectorXd> seeds, Eigen::VectorXd limits_lower, Eigen::VectorXd limits_upper,
                  int n_random, double duplicate_tol, int max_solutions, std::string solver_name)
    : inner_(std::move(inner))
    , seeds_(std::move(seeds))
    , lower_(std::move(limits_lower))
    , upper_(std::move(limits_upper))
    , n_random_(n_random)
    , duplicate_tol_(duplicate_tol)
    , max_solutions_(max_solutions)
    , solver_name_(std::move(solver_name))
  {
  }

  tesseract_kinematics::IKSolutions calcInvKin(const tesseract_common::TransformMap& tip_link_poses,
                                               const Eigen::Ref<const Eigen::VectorXd>& seed) const override
  {
    tesseract_kinematics::IKSolutions out;

    // Cap the number of solutions returned. Descartes builds a ladder graph with
    // one vertex per IK solution per sampled tool pose, and edges between EVERY
    // pair on consecutive rungs -- so memory grows with the square of this count.
    // Uncapped multi-seed IK drove the planning node to 28 GB and OOM-killed the
    // machine; a handful of diverse solutions gives the graph what it needs.
    auto collect = [&](const Eigen::VectorXd& s) {
      if (max_solutions_ > 0 && static_cast<int>(out.size()) >= max_solutions_)
        return;
      tesseract_kinematics::IKSolutions sols;
      try
      {
        sols = inner_->calcInvKin(tip_link_poses, s);
      }
      catch (const std::exception&)
      {
        return;  // a seed that throws is simply not useful
      }
      for (const Eigen::VectorXd& q : sols)
      {
        // Drop solutions outside the joint limits: the planner would reject them
        // anyway, and keeping them here would crowd out the usable ones.
        if (lower_.size() == q.size() && ((q.array() < lower_.array() - 1e-6).any() ||
                                          (q.array() > upper_.array() + 1e-6).any()))
          continue;
        bool duplicate = false;
        for (const Eigen::VectorXd& e : out)
        {
          if (e.size() == q.size() && ((e - q).cwiseAbs().maxCoeff() < duplicate_tol_))
          {
            duplicate = true;
            break;
          }
        }
        if (!duplicate)
          out.push_back(q);
        if (max_solutions_ > 0 && static_cast<int>(out.size()) >= max_solutions_)
          return;
      }
    };

    collect(seed);  // the caller's seed first, so its solution stays preferred
    for (const Eigen::VectorXd& s : seeds_)
      collect(s);

    if (n_random_ > 0 && lower_.size() == seed.size())
    {
      // Deterministic on purpose: planning must be reproducible run to run.
      std::mt19937 gen(42);
      for (int i = 0; i < n_random_; ++i)
      {
        Eigen::VectorXd s(lower_.size());
        for (Eigen::Index j = 0; j < lower_.size(); ++j)
        {
          std::uniform_real_distribution<double> d(lower_[j], upper_[j]);
          s[j] = d(gen);
        }
        collect(s);
      }
    }
    return out;
  }

  std::vector<std::string> getJointNames() const override { return inner_->getJointNames(); }
  Eigen::Index numJoints() const override { return inner_->numJoints(); }
  std::string getBaseLinkName() const override { return inner_->getBaseLinkName(); }
  std::string getWorkingFrame() const override { return inner_->getWorkingFrame(); }
  std::vector<std::string> getTipLinkNames() const override { return inner_->getTipLinkNames(); }
  std::string getSolverName() const override { return solver_name_; }

  tesseract_kinematics::InverseKinematics::UPtr clone() const override
  {
    return std::make_unique<MultiSeedInvKin>(inner_, seeds_, lower_, upper_, n_random_, duplicate_tol_,
                                             max_solutions_, solver_name_);
  }

private:
  std::shared_ptr<tesseract_kinematics::InverseKinematics> inner_;
  std::vector<Eigen::VectorXd> seeds_;
  Eigen::VectorXd lower_, upper_;
  int n_random_;
  double duplicate_tol_;
  int max_solutions_;
  std::string solver_name_;
};

template <typename T>
T get(const YAML::Node& node, const std::string& key)
{
  try
  {
    return node[key].as<T>();
  }
  catch (const YAML::Exception&)
  {
    std::stringstream ss;
    ss << "MultiSeedInvKin: failed to read '" << key << "' as " << boost::core::demangle(typeid(T).name());
    throw std::runtime_error(ss.str());
  }
}

class MultiSeedInvKinFactory : public tesseract_kinematics::InvKinFactory
{
public:
  tesseract_kinematics::InverseKinematics::UPtr create(const std::string& solver_name,
                                                       const tesseract_scene_graph::SceneGraph& scene_graph,
                                                       const tesseract_scene_graph::SceneState& /*scene_state*/,
                                                       const tesseract_kinematics::KinematicsPluginFactory& /*f*/,
                                                       const YAML::Node& config) const override
  {
    auto base_link = get<std::string>(config, "base_link");
    auto tip_link = get<std::string>(config, "tip_link");

    auto inner = std::make_shared<tesseract_kinematics::KDLInvKinChainLMA>(
        scene_graph, base_link, tip_link, tesseract_kinematics::KDLInvKinChainLMA::Config{},
        solver_name + "_inner");

    // Joint limits, in the solver's own joint order.
    const std::vector<std::string> joint_names = inner->getJointNames();
    Eigen::VectorXd lower(joint_names.size()), upper(joint_names.size());
    for (std::size_t i = 0; i < joint_names.size(); ++i)
    {
      const auto& joint = scene_graph.getJoint(joint_names[i]);
      lower[static_cast<Eigen::Index>(i)] = joint->limits->lower;
      upper[static_cast<Eigen::Index>(i)] = joint->limits->upper;
    }

    std::vector<Eigen::VectorXd> seeds;
    if (config["seeds"])
    {
      for (const YAML::Node& s : config["seeds"])
      {
        auto v = s.as<std::vector<double>>();
        if (v.size() != joint_names.size())
          throw std::runtime_error("MultiSeedInvKin: a seed has " + std::to_string(v.size()) + " values but the chain has " +
                                   std::to_string(joint_names.size()) + " joints");
        seeds.emplace_back(Eigen::Map<Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size())));
      }
    }
    if (seeds.empty())
      seeds.push_back(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_names.size())));
    // The mid-range posture is a good generic starting point away from limits.
    seeds.push_back(0.5 * (lower + upper));

    int n_random = config["n_random_seeds"] ? config["n_random_seeds"].as<int>() : 0;
    double dup = config["duplicate_tolerance"] ? config["duplicate_tolerance"].as<double>() : 1e-3;
    int max_sol = config["max_solutions"] ? config["max_solutions"].as<int>() : 4;

    return std::make_unique<MultiSeedInvKin>(inner, seeds, lower, upper, n_random, dup, max_sol, solver_name);
  }
};

}  // namespace snp_motion_planning

TESSERACT_ADD_PLUGIN(snp_motion_planning::MultiSeedInvKinFactory, MultiSeedInvKinFactory)
