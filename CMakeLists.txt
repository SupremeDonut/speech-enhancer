cmake_minimum_required(VERSION 3.20)
project(speech_isolation LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This runtime currently targets Windows x64 only")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(KISSFFT_STATIC ON CACHE BOOL "Build KissFFT statically" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "No upstream tests" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "No upstream tools" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "No pkg-config" FORCE)
set(KISSFFT_DATATYPE float CACHE STRING "Float DSP" FORCE)
FetchContent_Declare(kissfft GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG e5e3fac46e0d94a8f8170c06706b7a4218828333 GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(kissfft)

FetchContent_Declare(miniaudio GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG f40cf03f80cdb7e741d43e53b7e706e8c1394bcf GIT_SHALLOW FALSE)
set(MINIAUDIO_BUILD_EXAMPLES OFF CACHE BOOL "No miniaudio examples" FORCE)
set(MINIAUDIO_BUILD_TESTS OFF CACHE BOOL "No miniaudio tests" FORCE)
set(MINIAUDIO_BUILD_TOOLS OFF CACHE BOOL "No miniaudio tools" FORCE)
set(MINIAUDIO_NO_DECODING ON CACHE BOOL "No decoding APIs" FORCE)
set(MINIAUDIO_NO_ENCODING ON CACHE BOOL "No encoding APIs" FORCE)
set(MINIAUDIO_NO_RESOURCE_MANAGER ON CACHE BOOL "No resource manager" FORCE)
set(MINIAUDIO_NO_NODE_GRAPH ON CACHE BOOL "No node graph" FORCE)
set(MINIAUDIO_NO_EXTRA_NODES ON CACHE BOOL "No extra miniaudio graph nodes" FORCE)
set(MINIAUDIO_NO_ENGINE ON CACHE BOOL "No miniaudio engine" FORCE)
FetchContent_MakeAvailable(miniaudio)

set(ORT_VERSION 1.22.1)
set(ORT_ROOT "${CMAKE_BINARY_DIR}/_deps/onnxruntime-win-x64-${ORT_VERSION}")
set(ORT_ARCHIVE "${CMAKE_BINARY_DIR}/_deps/onnxruntime-win-x64-${ORT_VERSION}.zip")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
    file(DOWNLOAD
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip"
        "${ORT_ARCHIVE}"
        EXPECTED_HASH SHA256=855276cd4be3cda14fe636c69eb038d75bf5bcd552bda1193a5d79c51f436dfe
        SHOW_PROGRESS STATUS ORT_DOWNLOAD_STATUS)
    list(GET ORT_DOWNLOAD_STATUS 0 ORT_DOWNLOAD_CODE)
    if(NOT ORT_DOWNLOAD_CODE EQUAL 0)
        list(GET ORT_DOWNLOAD_STATUS 1 ORT_DOWNLOAD_ERROR)
        message(FATAL_ERROR "Failed downloading ONNX Runtime SDK: ${ORT_DOWNLOAD_ERROR}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${ORT_ARCHIVE}" DESTINATION "${CMAKE_BINARY_DIR}/_deps"
        PATTERNS "*/include/*" "*/lib/onnxruntime.lib" "*/lib/onnxruntime.dll")
endif()
set(ORT_INCLUDE "${ORT_ROOT}/include")
set(ORT_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib")
set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
if(NOT EXISTS "${ORT_IMPLIB}" OR NOT EXISTS "${ORT_DLL}")
    message(FATAL_ERROR "ONNX Runtime ${ORT_VERSION} Windows x64 SDK is incomplete")
endif()
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB "${ORT_IMPLIB}"
    IMPORTED_LOCATION "${ORT_DLL}" INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE}")

add_library(speech_dsp STATIC src/DSPPipeline.cpp)
target_include_directories(speech_dsp PUBLIC include)
target_link_libraries(speech_dsp PUBLIC kissfft::kissfft)

add_library(speech_runtime STATIC src/InferenceSession.cpp src/WavFile.cpp src/AudioEngine.cpp)
target_include_directories(speech_runtime PUBLIC include)
target_link_libraries(speech_runtime PUBLIC speech_dsp onnxruntime miniaudio)
if(WIN32)
    target_link_libraries(speech_runtime PUBLIC ole32 uuid)
endif()

add_executable(speech_isolation src/main.cpp)
target_link_libraries(speech_isolation PRIVATE speech_runtime)
add_custom_command(TARGET speech_isolation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" "$<TARGET_FILE_DIR:speech_isolation>/onnxruntime.dll")

include(CTest)
if(BUILD_TESTING)
    add_executable(dsp_tests tests/dsp_tests.cpp)
    target_link_libraries(dsp_tests PRIVATE speech_dsp)
    add_test(NAME dsp_identity_mask_reset COMMAND dsp_tests)
    add_executable(runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(runtime_tests PRIVATE speech_runtime)
    add_test(NAME runtime_wav_spsc_ort COMMAND runtime_tests)
    add_test(NAME cli_system_audio_requires_both_devices
        COMMAND ${CMAKE_COMMAND} "-DPROGRAM=$<TARGET_FILE:speech_isolation>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_system_audio_cli.cmake")
endif()
