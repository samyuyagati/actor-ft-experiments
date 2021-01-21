#include <list>

#include "src/ray/protobuf/core_worker.pb.h"
#include "ray/common/id.h"
#include "ray/core_worker/context.h"

namespace ray {

class TaskLogger {
 public:
  TaskLogger() {}

  void LogRequest(const rpc::PushTaskRequest &request, const WorkerID worker_id);

  // void LogTime(const ActorID actor_id, int duration);

  // int GetTotalDuration()

 private:
  std::list<const rpc::PushTaskRequest> log_;
};

}  // namespace ray
