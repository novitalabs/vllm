# Novita fused kernels:
#   1) AllReduce + RMSNorm + Residual Add (+ optional quant)
#   2) QK RMSNorm + TP variance exchange
#   3) QK-Norm + RoPE + FP8 quant + KV cache store (with TP)

set(NOVITA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/allreduce_fusion_wrapper.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/qk_rmsnorm_tp_wrapper.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/fused_qknorm_rope_fp8_kvstore_kernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/torch_bindings.cpp")

set(NOVITA_GPU_FLAGS ${VLLM_GPU_FLAGS})
list(APPEND NOVITA_GPU_FLAGS "--use_fast_math")

# Novita kernels target H200 (sm_90) only.
set(NOVITA_GPU_ARCHES "90")

define_extension_target(
    _novita_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${NOVITA_SRCS}
    COMPILE_FLAGS ${NOVITA_GPU_FLAGS}
    ARCHITECTURES ${NOVITA_GPU_ARCHES}
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita
    USE_SABI 3
    WITH_SOABI)
