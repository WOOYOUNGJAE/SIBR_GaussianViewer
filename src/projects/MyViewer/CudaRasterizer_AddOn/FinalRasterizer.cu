#include "FinalRasterizer.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"
#include "FinalRasterizer.h"

#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


namespace cg = cooperative_groups;

#include "auxiliary.h"

#include "My_Utils.h"

// Cuda Functions Defines
#include "helper_math.h"

// My Defines
#define CHECK_NORMAL_COLOR 0
namespace CudaRasterizer
{
	namespace DF_L
	{
		// Forward method for converting the input spherical harmonics
		// coefficients of each Gaussian to a simple RGB color.
		__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped);

		// Forward version of 2D covariance matrix computation
		__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix);

		// Forward method for converting scale and rotation properties of each
		// Gaussian to a 3D covariance matrix in world space. Also takes care
		// of quaternion normalization.
		__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D);
	}
}

using namespace CudaRasterizer;

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int2* rects)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		if (rects == nullptr)
			getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		else
			getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
		if (idx == L - 1)
			ranges[currtile].y = L;
	}
}

CudaRasterizer::DF_L::GeometryState CudaRasterizer::DF_L::GeometryState::fromChunk(char*& chunk, size_t P)
{
	DF_L::GeometryState geom;
	DF_L::obtain(chunk, geom.depths, P, 128);
	DF_L::obtain(chunk, geom.clamped, P * 3, 128);
	DF_L::obtain(chunk, geom.internal_radii, P, 128);
	DF_L::obtain(chunk, geom.means2D, P, 128);
	DF_L::obtain(chunk, geom.cov3D, P * 6, 128);
	DF_L::obtain(chunk, geom.conic_opacity, P, 128);
	DF_L::obtain(chunk, geom.rgb, P * 3, 128);
	DF_L::obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	DF_L::obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	DF_L::obtain(chunk, geom.point_offsets, P, 128);
	DF_L::obtain(chunk, geom.normal, P, 128);
	return geom;
}

CudaRasterizer::DF_L::ImageState CudaRasterizer::DF_L::ImageState::fromChunk(char*& chunk, size_t N)
{
	DF_L::ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::DF_L::BinningState CudaRasterizer::DF_L::BinningState::fromChunk(char*& chunk, size_t P)
{
	DF_L::BinningState binning;
	DF_L::obtain(chunk, binning.point_list, P, 128);
	DF_L::obtain(chunk, binning.point_list_unsorted, P, 128);
	DF_L::obtain(chunk, binning.point_list_keys, P, 128);
	DF_L::obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::DF_L::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
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
	int* radii,
	int* rects,
	float* boxmin,
	float* boxmax,
	float* normalsCuda,
	float* out_pixelNormals, LIGHT_DESC light)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = DF_L::required<DF_L::GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	DF_L::GeometryState geomState = DF_L::GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	int img_chunk_size = DF_L::required<DF_L::ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	DF_L::ImageState imgState = DF_L::ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	float3 minn = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	float3 maxx = { FLT_MAX, FLT_MAX, FLT_MAX };
	if (boxmin != nullptr)
	{
		minn = *((float3*)boxmin);
		maxx = *((float3*)boxmax);
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	DF_L::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		(int2*)rects,
		minn,
		maxx,
		normalsCuda,
		geomState.normal
	);

	uint32_t* check_uint32_t2 = new uint32_t[P];
	DEBUG_WATCH_CUDA_MEM(check_uint32_t2, geomState.tiles_touched, sizeof(uint32_t) * P);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);


	uint32_t* check_uint32_t = new uint32_t[P];
	DEBUG_WATCH_CUDA_MEM(check_uint32_t, geomState.point_offsets, sizeof(uint32_t) * P);


	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	if (num_rendered == 0)
		return 0;
	//num_rendered -= 1;
	int binning_chunk_size = DF_L::required<DF_L::BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	DF_L::BinningState binningState = DF_L::BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// KEY: [tile ID|depth]
	// VALUE: Gaussian ID
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		(int2*)rects
		);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges
		);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	DF_L::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.normal,
		normalsCuda, out_pixelNormals, light);



	return num_rendered;
}

namespace CudaRasterizer
{
	namespace  DF_L
	{

		// Perform initial steps for each Gaussian prior to rasterization.
		template<int C>
		__global__ void preprocessCUDA(int P, int D, int M,
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
			const float tan_fovx, float tan_fovy,
			const float focal_x, float focal_y,
			int* radii,
			float2* points_xy_image,
			float* depths,
			float* cov3Ds,
			float* rgb,
			float4* conic_opacity,
			const dim3 grid,
			uint32_t* tiles_touched,
			bool prefiltered,
			int2* rects,
			float3 boxmin,
			float3 boxmax,
			float* gaussianNormalsCuda,
			float3* pOutNormals)
		{
			auto idx = cg::this_grid().thread_rank();
			if (idx >= P)
				return;

			// Initialize radius and touched tiles to 0. If this isn't changed,
			// this Gaussian will not be processed further.
			radii[idx] = 0;
			tiles_touched[idx] = 0;

			// Perform near culling, quit if outside.
			float3 p_view;
			if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
				return;

			// Transform point by projecting
			float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

			if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
				p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
				return;

			float4 p_hom = transformPoint4x4(p_orig, projmatrix);
			float p_w = 1.0f / (p_hom.w + 0.0000001f);
			float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

			// If 3D covariance matrix is precomputed, use it, otherwise compute
			// from scaling and rotation parameters. 
			const float* cov3D;
			if (cov3D_precomp != nullptr)
			{
				cov3D = cov3D_precomp + idx * 6;
			}
			else
			{
				DF_L::computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
				cov3D = cov3Ds + idx * 6;
			}

			// Compute 2D screen-space covariance matrix
			float3 cov = DF_L::computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

			// Invert covariance (EWA algorithm)
			float det = (cov.x * cov.z - cov.y * cov.y);
			if (det == 0.0f)
				return;
			float det_inv = 1.f / det;
			float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

			// Compute extent in screen space (by finding eigenvalues of
			// 2D covariance matrix). Use extent to compute a bounding rectangle
			// of screen-space tiles that this Gaussian overlaps with. Quit if
			// rectangle covers 0 tiles. 

			float mid = 0.5f * (cov.x + cov.z);
			float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
			float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
			float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
			float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
			uint2 rect_min, rect_max;

			if (rects == nullptr) 	// More conservative
			{
				getRect(point_image, my_radius, rect_min, rect_max, grid);
			}
			else // Slightly more aggressive, might need a math cleanup
			{
				const int2 my_rect = { (int)ceil(3.f * sqrt(cov.x)), (int)ceil(3.f * sqrt(cov.z)) };
				rects[idx] = my_rect;
				getRect(point_image, my_rect, rect_min, rect_max, grid);
			}

			if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
				return;

			// If colors have been precomputed, use them, otherwise convert
			// spherical harmonics coefficients to RGB color.
			if (colors_precomp == nullptr)
			{
				glm::vec3 result = DF_L::computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
				rgb[idx * C + 0] = result.x;
				rgb[idx * C + 1] = result.y;
				rgb[idx * C + 2] = result.z;
				/*rgb[idx * C + 0] = gaussianNormalsCuda[idx * 3 + 0];
				rgb[idx * C + 1] = gaussianNormalsCuda[idx * 3 + 1];
				rgb[idx * C + 2] = gaussianNormalsCuda[idx * 3 + 2];*/
			}

			// Store some useful helper data for the next steps.
			depths[idx] = p_view.z;
			radii[idx] = my_radius;
			points_xy_image[idx] = point_image;
			// Inverse 2D covariance and opacity neatly pack into one float4
			conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
			tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

			// Store normals
			// TODO : normal projection º¯È¯
			pOutNormals[idx].x = gaussianNormalsCuda[idx * 3 + 0];
			pOutNormals[idx].y = gaussianNormalsCuda[idx * 3 + 1];
			pOutNormals[idx].z = gaussianNormalsCuda[idx * 3 + 2];
		}

		// Main rasterization method. Collaboratively works on one tile per
		// block, each thread treats one pixel. Alternates between fetching 
		// and rasterizing data.
		template <uint32_t CHANNELS>
		__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
			renderCUDA(
				const uint2* __restrict__ ranges,
				const uint32_t* __restrict__ point_list,
				int W, int H,
				const float2* __restrict__ points_xy_image,
				const float* __restrict__ features,
				const float4* __restrict__ conic_opacity,
				float* __restrict__ final_T,
				uint32_t* __restrict__ n_contrib,
				const float* __restrict__ bg_color,
				float* __restrict__ out_color,
				float3* preprocessedNormals,
				LIGHT_DESC light)
		{

			// Identify current tile and associated min/max pixel range.
			auto block = cg::this_thread_block();
			uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
			uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
			uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
			uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
			uint32_t pix_id = W * pix.y + pix.x;
			float2 pixf = { (float)pix.x, (float)pix.y };

			// Check if this thread is associated with a valid pixel or outside.
			bool inside = pix.x < W&& pix.y < H;
			// Done threads can help with fetching, but don't rasterize
			bool done = !inside;

			// Load start/end range of IDs to process in bit sorted list.
			uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
			const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
			int toDo = range.y - range.x;

			// Allocate storage for batches of collectively fetched data.
			__shared__ int collected_id[BLOCK_SIZE];
			__shared__ float2 collected_xy[BLOCK_SIZE];
			__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

			// Initialize helper variables
			float T = 1.0f;
			uint32_t contributor = 0;
			uint32_t last_contributor = 0;
			float C[CHANNELS] = { 0 };
			float3 curPixNormal{};
			// Iterate over batches until all done or range is complete
			for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
			{
				// End if entire block votes that it is done rasterizing
				int num_done = __syncthreads_count(done);
				if (num_done == BLOCK_SIZE)
					break;

				// Collectively fetch per-Gaussian data from global to shared
				int progress = i * BLOCK_SIZE + block.thread_rank();
				if (range.x + progress < range.y)
				{
					int coll_id = point_list[range.x + progress];
					collected_id[block.thread_rank()] = coll_id;
					collected_xy[block.thread_rank()] = points_xy_image[coll_id];
					collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				}
				block.sync();

				// Iterate over current batch
				for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
				{
					// Keep track of current position in range
					contributor++;

					// Resample using conic matrix (cf. "Surface 
					// Splatting" by Zwicker et al., 2001)
					float2 xy = collected_xy[j];
					float2 d = { xy.x - pixf.x, xy.y - pixf.y };
					float4 con_o = collected_conic_opacity[j];
					float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
					if (power > 0.0f)
						continue;

					// Eq. (2) from 3D Gaussian splatting paper.
					// Obtain alpha by multiplying with Gaussian opacity
					// and its exponential falloff from mean.
					// Avoid numerical instabilities (see paper appendix). 
					float alpha = min(0.99f, con_o.w * exp(power));
					if (alpha < 1.0f / 255.0f)
						continue;
					float test_T = T * (1 - alpha);
					if (test_T < 0.0001f)
					{
						done = true;
						continue;
					}

					// Eq. (3) from 3D Gaussian splatting paper.
					for (int ch = 0; ch < CHANNELS; ch++)
					{
#if CHECK_NORMAL_COLOR 1
#else
						curPixNormal += preprocessedNormals[collected_id[j]] * alpha * T;
						C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
#endif
					}

					T = test_T;


					// Keep track of last range entry to update this
					// pixel.
					last_contributor = contributor;


				}
			}

			// All threads that treat valid pixel write out their final
			// rendering data to the frame and auxiliary buffers.
			if (inside)
			{
				final_T[pix_id] = T;
				n_contrib[pix_id] = last_contributor;
				//// Visualize Pixel Normals rgb
				{
#if CHECK_NORMAL_COLOR
					curPixNormal = normalize(curPixNormal);
					C[0] = curPixNormal.x;
					C[1] = curPixNormal.y;
					C[2] = curPixNormal.z;
#endif
					for (int ch = 0; ch < CHANNELS; ch++)
					{
						out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
					}

					float3 curPixelColor = { C[0] + T * bg_color[0],C[1] + T * bg_color[1],C[2] + T * bg_color[2], };

					// deferred Rendering Pass 2
					// apply dir_diffuse color
					curPixNormal = normalize(curPixNormal);
					float3 lightDirMinus = make_float3(-light.dir_Dir.x(), -light.dir_Dir.y(), -light.dir_Dir.z());
					float lambertPower = ::max(dot(curPixNormal, lightDirMinus), 0.f);

					float3 ambientColor = make_float3(light.ambient.x(), light.ambient.y(), light.ambient.z()) * light.ambientPower;

					float3 lightDiffuseColor = { light.dir_diffuse.x(), light.dir_diffuse.y(), light.dir_diffuse.z() };
					float3 finalColor = curPixelColor * lightDiffuseColor * lambertPower + ambientColor;

					out_color[0 * H * W + pix_id] = finalColor.x;
					out_color[1 * H * W + pix_id] = finalColor.y;
					out_color[2 * H * W + pix_id] = finalColor.z;

				}
			}
		}
	}



	void DF_L::render(const dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list, int W, int H,
		const float2* points_xy_image, const float* features, const float4* conic_opacity, float* final_T,
		uint32_t* n_contrib, const float* bg_color, float* out_color, float3* preprocessedNormals, float* gsNormals, float* out_pixelNormals, LIGHT_DESC light)
	{
		// Pass 1
		DF_L::renderCUDA<NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			points_xy_image, // means2D
			features, // colors
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			preprocessedNormals, light);

	}

	void DF_L::preprocess(int P, int D, int M,
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
		float3* pOutNormals)
	{
		DF_L::preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
			P, D, M,
			orig_points, //means3D
			scales,
			scale_modifier,
			rotations,
			opacities,
			shs,
			clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix,
			projmatrix,
			cam_pos,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			radii,
			points_xy_image, // means2D
			depths,
			cov3Ds,
			colors, // rgb
			conic_opacity,
			grid,
			tiles_touched,
			prefiltered,
			rects,
			boxmin,
			boxmax,
			gaussianNormalsCuda,
			pOutNormals
			);
	}



	__device__ glm::vec3 DF_L::computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
	{
		// The implementation is loosely based on code for 
		// "Differentiable Point-Based Radiance Fields for 
		// Efficient View Synthesis" by Zhang et al. (2022)
		glm::vec3 pos = means[idx];
		glm::vec3 dir = pos - campos;
		dir = dir / glm::length(dir);

		glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
		glm::vec3 result = SH_C0 * sh[0];

		if (deg > 0)
		{
			float x = dir.x;
			float y = dir.y;
			float z = dir.z;
			result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

			if (deg > 1)
			{
				float xx = x * x, yy = y * y, zz = z * z;
				float xy = x * y, yz = y * z, xz = x * z;
				result = result +
					SH_C2[0] * xy * sh[4] +
					SH_C2[1] * yz * sh[5] +
					SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					SH_C2[3] * xz * sh[7] +
					SH_C2[4] * (xx - yy) * sh[8];

				if (deg > 2)
				{
					result = result +
						SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						SH_C3[1] * xy * z * sh[10] +
						SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						SH_C3[5] * z * (xx - yy) * sh[14] +
						SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
				}
			}
		}
		result += 0.5f;

		// RGB colors are clamped to positive values. If values are
		// clamped, we need to keep track of this for the backward pass.
		clamped[3 * idx + 0] = (result.x < 0);
		clamped[3 * idx + 1] = (result.y < 0);
		clamped[3 * idx + 2] = (result.z < 0);
		return glm::max(result, 0.0f);
	}

	__device__ float3 DF_L::computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
	{
		// The following models the steps outlined by equations 29
		// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
		// Additionally considers aspect / scaling of viewport.
		// Transposes used to account for row-/column-major conventions.
		float3 t = transformPoint4x3(mean, viewmatrix);

		const float limx = 1.3f * tan_fovx;
		const float limy = 1.3f * tan_fovy;
		const float txtz = t.x / t.z;
		const float tytz = t.y / t.z;
		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;

		glm::mat3 J = glm::mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0);

		glm::mat3 W = glm::mat3(
			viewmatrix[0], viewmatrix[4], viewmatrix[8],
			viewmatrix[1], viewmatrix[5], viewmatrix[9],
			viewmatrix[2], viewmatrix[6], viewmatrix[10]);

		glm::mat3 T = W * J;

		glm::mat3 Vrk = glm::mat3(
			cov3D[0], cov3D[1], cov3D[2],
			cov3D[1], cov3D[3], cov3D[4],
			cov3D[2], cov3D[4], cov3D[5]);

		glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

		// Apply low-pass filter: every Gaussian should be at least
		// one pixel wide/high. Discard 3rd row and column.
		cov[0][0] += 0.3f;
		cov[1][1] += 0.3f;
		return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
	}

	__device__ void DF_L::computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
	{
		// Create scaling matrix
		glm::mat3 S = glm::mat3(1.0f);
		S[0][0] = mod * scale.x;
		S[1][1] = mod * scale.y;
		S[2][2] = mod * scale.z;

		// Normalize quaternion to get valid rotation
		glm::vec4 q = rot;// / glm::length(rot);
		float r = q.x;
		float x = q.y;
		float y = q.z;
		float z = q.w;

		// Compute rotation matrix from quaternion
		glm::mat3 R = glm::mat3(
			1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
			2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
			2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
		);

		glm::mat3 M = S * R;

		// Compute 3D world covariance matrix Sigma
		glm::mat3 Sigma = glm::transpose(M) * M;

		// Covariance is symmetric, only store upper right
		cov3D[0] = Sigma[0][0];
		cov3D[1] = Sigma[0][1];
		cov3D[2] = Sigma[0][2];
		cov3D[3] = Sigma[1][1];
		cov3D[4] = Sigma[1][2];
		cov3D[5] = Sigma[2][2];
	}
}