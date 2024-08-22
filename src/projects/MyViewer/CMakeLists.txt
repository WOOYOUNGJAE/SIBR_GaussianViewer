set(SIBR_PROJECT "MyViewer")
project(sibr_MyViewer_all)

add_subdirectory(apps)
add_subdirectory(renderer)

include(install_runtime)
subdirectory_target(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR} "projects/${SIBR_PROJECT}")
