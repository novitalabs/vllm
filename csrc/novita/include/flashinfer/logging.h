// No-op logging stub - the allreduce fusion kernel includes this header
// but does not use any logging macros. This avoids a spdlog dependency.
#ifndef FLASHINFER_LOGGING_H_
#define FLASHINFER_LOGGING_H_

#define FLASHINFER_LOG_TRACE(...)
#define FLASHINFER_LOG_DEBUG(...)
#define FLASHINFER_LOG_INFO(...)
#define FLASHINFER_LOG_WARN(...)
#define FLASHINFER_LOG_ERROR(...)
#define FLASHINFER_LOG_CRITICAL(...)

#endif  // FLASHINFER_LOGGING_H_
