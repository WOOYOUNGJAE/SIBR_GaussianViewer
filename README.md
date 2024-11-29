# SIBR_GaussianViewer
This repository is an extended work of the SIBR Gaussian Viewer.

This project is based on the work from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Original Readme
https://gitlab.inria.fr/sibr/sibr_core/-/blob/develop/README.md?ref_type=heads

## Working Project
/src/projects/MyViewer

## 필수 요구 사항

- **CUDA Toolkit 11.8**
- **Visual Studio 2019 Build Tools** - IDE도 2019로 통일하는 것을 강력히 권장

## How To Build and Run
  
1. CMake로 VS 2019 Build Tool을 지정한 후 솔루션과 프로젝트 생성 (Configure -> Generate)
   - **Where is the source code**: `C:/Users/../SIBR_GaussianViewer`
   - **Where to build the binaries**: `C:/Users/../SIBR_GaussianViewer/build`
2. Copy_AddOn_To_extlibs.bat 실행

3. `build/sibr_projects.sln` 실행
 
4. extlibes/CudaRastizer 프로젝트에 아래 파일들 추가 (프로젝트 우클릭 -> Add -> Existing Item)
   - extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/FinalRasterizer.h
   - extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/FinalRasterizer.cu
   - extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/My_Utils.h
   - extlibs/CudaRasterizer/CudaRasterizer/cuda_rasterizer/helper_math.h
   
5. `ALL_BUILD` 프로젝트를 지정한 후 빌드

6. `INSTALL` 프로젝트를 지정한 후 빌드

7. `sibr_MyViewer_app` 프로젝트를 Set as Startup Project로 지정한 후 빌드

8. VS에서 디버깅 할 경우:
   - `sibr_MyViewer_app → properties → Configuration Properties → Debugging → Command Arguments`에 `-m ..\..\Assets` 추가
   
## 기타
src/projects/MyViewer/apps/MyViewerApp/MyInclude.h 의 USE_MESHRENDERER 로 Deferred Rendering 여부 결정
