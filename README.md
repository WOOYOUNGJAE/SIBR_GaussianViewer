# SIBR_GaussianViewer
This repository is an extended work of the SIBR Gaussian Viewer.

This project is based on the work from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Original Readme
https://gitlab.inria.fr/sibr/sibr_core/-/blob/develop/README.md?ref_type=heads

## 필수 요구 사항

- **CUDA Toolkit 11.8**
- **Visual Studio 2019 Build Tools** - IDE도 2019로 통일하는 것을 권장

## How To Build

1. CMake로 VS 2019 Build Tool을 지정한 후 솔루션과 프로젝트 생성
   - **Where is the source code**: `C:/Users/../SIBR_GaussianViewer`
   - **Where to build the binaries**: `C:/Users/../SIBR_GaussianViewer/build`
   
2. Copy_AddOn_To_extlibs.bat 실행

3. `build/sibr_projects.sln` 실행 후, `extlibs/CudaRasterizer` 프로젝트에 다음 파일들을 추가 (Add Existing Item):
   - `FinalRasterizer.cu`
   - `FinalRasterizer.h`
   - `helper_math.h`
   - `My_Utils.h`
   
4. `ALL_BUILD` 프로젝트를 지정한 후 빌드

5. `INSTALL` 프로젝트를 지정한 후 빌드

6. `sibr_MyViewer_app` 프로젝트를 지정한 후 빌드

7. VS에서 디버깅 할 경우:
   - `sibr_MyViewer_app → properties → Configuration Properties → Debugging → Command Arguments`에 `-m ..\..\Assets` 추가

## Assets 경로
Root
├── Assets

│   └──  OnlyPly # 이 안에 cameras.json과 ply, ply는 point_cloud.ply 단 한 개로 고정 (24.08 기준)
│…
└── src