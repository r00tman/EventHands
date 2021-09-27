############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

# xtensor-blas cmake module
# This module sets the following variables in your project::
#
#   xtensor_blas_FOUND - true if xtensor-blas found on the system
#   xtensor_blas_INCLUDE_DIR - the directory containing xtensor-blas headers
#   xtensor_blas_LIBRARY - empty


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was xtensor-blasConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET xtensor-blas)
  include("${CMAKE_CURRENT_LIST_DIR}/xtensor-blasTargets.cmake")
  get_target_property(xtensor-blas_INCLUDE_DIRS xtensor-blas INTERFACE_INCLUDE_DIRECTORIES)
  find_dependency(BLAS REQUIRED)
  find_dependency(LAPACK REQUIRED)
  target_link_libraries(xtensor-blas INTERFACE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.8)
    target_compile_features(xtensor-blas INTERFACE cxx_std_14)
  endif()
endif()

set(PN xtensor_blas)
set_and_check(TEMP_XTENSOR_BLAS_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")
set(${PN}_INCLUDE_DIRS "${TEMP_XTENSOR_BLAS_INCLUDE_DIR}")
unset(TEMP_XTENSOR_BLAS_INCLUDE_DIR)
set(${PN}_LIBRARY "")
check_required_components(${PN})
