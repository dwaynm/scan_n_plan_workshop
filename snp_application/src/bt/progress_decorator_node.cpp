#include <snp_application/bt/progress_decorator_node.h>
#include <snp_application/bt/utils.h>

#include <QProgressBar>

namespace snp_application
{
BT::NodeStatus ProgressDecoratorNode::tick()
{
  auto progress_bar = config().blackboard->get<QProgressBar*>(PROGRESS_BAR_KEY);
  auto start = getBTInput<int>(this, START_PORT_KEY);
  auto end = getBTInput<int>(this, END_PORT_KEY);

  // Guard against a null/dangling progress bar pointer (e.g. blackboard key not
  // remapped into this subtree) — invokeMethod on a bad QObject* segfaults.
  // The progress bar is purely cosmetic, so skipping the update is safe.
  // Set the initial progress
  if (progress_bar)
    QMetaObject::invokeMethod(progress_bar, "setValue", Qt::QueuedConnection, Q_ARG(int, start));

  BT::NodeStatus status = child()->executeTick();
  switch (status)
  {
    case BT::NodeStatus::SUCCESS:
      // Set the final progress
      if (progress_bar)
        QMetaObject::invokeMethod(progress_bar, "setValue", Qt::QueuedConnection, Q_ARG(int, end));
      break;
    default:
      break;
  }

  return status;
}

}  // namespace snp_application
