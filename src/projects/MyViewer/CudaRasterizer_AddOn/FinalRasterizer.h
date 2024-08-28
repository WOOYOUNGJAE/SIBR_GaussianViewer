#ifndef CUDA_FINAL_RASTERIZER_H_INCLUDED
#define CUDA_FINAL_RASTERIZER_H_INCLUDED

#include "rasterizer.h"

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

#include "../../../../extlibs/CudaRasterizer/CudaRasterizer/third_party/glm/glm/glm.hpp"

#include "../../../../src/projects/MyViewer/renderer/MyStructs.h"
namespace CudaRasterizer
{
	namespace DF_L
	{
		// Only for App's Final Rendering
		class Rasterizer
		{
		public:
			static int forward(
				std::function<char* (size_t)> geometryBuffer,
				std::function<char* (size_t)> binningBuffer,
				std::function<char* (size_t)> imageBuffer,
				const int P, int D, int M,
				const float* background,
				const float* bg_precomp,
				const int width, int height,
				const float* means3D,
				const float* shs,
				const float* colors_precomp,
				const float* opacities,
				const float* scales,
				const float scale_modifier,
				const float* rotations,
				const float* cov3D_precomp,
				const float* viewmatrix,
				const float* projmatrix,
				const float* cam_pos,
				const float tan_fovx, float tan_fovy,
				const bool prefiltered,
				float* out_color,
				int* radii = nullptr,
				int* rects = nullptr,
				float* boxmin = nullptr,
				float* boxmax = nullptr, float* normalsCuda = nullptr, float* out_pixelNormals = nullptr, LIGHT_DESC light = LIGHT_DESC());

			static int Read_GLTexture(
				const int width, int height,
				float* out_color, cudaArray** pCudaArr);

		};

		template <typename T>
		static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
		{
			std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
			ptr = reinterpret_cast<T*>(offset);
			chunk = reinterpret_cast<char*>(ptr + count);
		}

		struct GeometryState
		{
			size_t scan_size;
			float* depths;
			char* scanning_space;
			bool* clamped;
			int* internal_radii;
			float2* means2D;
			float* cov3D;
			float4* conic_opacity;
			float* rgb;
			uint32_t* point_offsets;
			uint32_t* tiles_touched;
			float3* normal;

			static GeometryState fromChunk(char*& chunk, size_t P);
		};


		struct ImageState
		{
			uint2* ranges;
			uint32_t* n_contrib;
			float* accum_alpha;

			static ImageState fromChunk(char*& chunk, size_t N);
		};

		struct BinningState
		{
			size_t sorting_size;
			uint64_t* point_list_keys_unsorted;
			uint64_t* point_list_keys;
			uint32_t* point_list_unsorted;
			uint32_t* point_list;
			char* list_sorting_space;

			static BinningState fromChunk(char*& chunk, size_t P);
		};

		template<typename T>
		size_t required(size_t P)
		{
			char* size = nullptr;
			T::fromChunk(size, P);
			return ((size_t)size) + 128;
		}

		// Perform initial steps for each Gaussian prior to rasterization.
		void preprocess(int P, int D, int M,
		                const float* orig_points,
		                const glm::vec3* scales,
		                const float scale_modifier,
		                const glm::vec4* rotations,
		                const float* opacities,
		                const float* shs,
		                bool* clamped,
		                const float* cov3D_precomp,
		                const float* colors_precomp,
		                const float* viewmatrix,
		                const float* projmatrix,
		                const glm::vec3* cam_pos,
		                const int W, int H,
		                const float focal_x, float focal_y,
		                const float tan_fovx, float tan_fovy,
		                int* radii,
		                float2* points_xy_image,
		                float* depths,
		                float* cov3Ds,
		                float* colors,
		                float4* conic_opacity,
		                const dim3 grid,
		                uint32_t* tiles_touched,
		                bool prefiltered,
		                int2* rects,
		                float3 boxmin,
		                float3 boxmax,
		                float* gaussianNormalsCuda,
		                float3* pOutNormals);

		// Main rasterization method.
		void render(
			const dim3 grid, dim3 block,
			const uint2* ranges,
			const uint32_t* point_list,
			int W, int H,
			const float2* points_xy_image,
			const float* features,
			const float4* conic_opacity,
			float* final_T,
			uint32_t* n_contrib,
			const float* bg_color,
			const float* bg_precomp,
			float* out_color, float3* preprocessedNormals, float* gsNormals, float* out_pixelNormals, LIGHT_DESC light);


	}
}

#endif