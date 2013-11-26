###############################################################################
#
# Build script for project
#
###############################################################################


set(CUFILES
  parallel_QS.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})

cuda_add_executable(parallel_QS  ${CUFILES})

target_link_libraries(parallel_QS cudpp)
