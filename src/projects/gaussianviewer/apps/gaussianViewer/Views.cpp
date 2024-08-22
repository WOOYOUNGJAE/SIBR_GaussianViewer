#include "Views.hpp"

#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include "FinalRasterizer.h"
#include <imgui_internal.h>

#pragma region ReDefine
#include "Scene.hpp"
#include "../../renderer/GaussianSurfaceRenderer.hpp"
#include "core/graphics/Texture.hpp"
namespace sibr {

	GaussianData::GaussianData(int num_gaussians, float* mean_data, float* rot_data, float* scale_data, float* alpha_data, float* color_data)
	{
		_num_gaussians = num_gaussians;
		glCreateBuffers(1, &meanBuffer);
		glCreateBuffers(1, &rotBuffer);
		glCreateBuffers(1, &scaleBuffer);
		glCreateBuffers(1, &alphaBuffer);
		glCreateBuffers(1, &colorBuffer);
		glNamedBufferStorage(meanBuffer, num_gaussians * 3 * sizeof(float), mean_data, 0);
		glNamedBufferStorage(rotBuffer, num_gaussians * 4 * sizeof(float), rot_data, 0);
		glNamedBufferStorage(scaleBuffer, num_gaussians * 3 * sizeof(float), scale_data, 0);
		glNamedBufferStorage(alphaBuffer, num_gaussians * sizeof(float), alpha_data, 0);
		glNamedBufferStorage(colorBuffer, num_gaussians * sizeof(float) * 48, color_data, 0);
	}

	void GaussianData::render(int G) const
	{
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, meanBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rotBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, scaleBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, alphaBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, colorBuffer);
		glDrawArraysInstanced(GL_TRIANGLES, 0, 36, G);
	}

	GaussianSurfaceRenderer::GaussianSurfaceRenderer(void)
	{
		_shader.init("GaussianSurface",
			sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/gaussian_surface.vert"),
			sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/gaussian_surface.frag"));

		_paramCamPos.init(_shader, "rayOrigin");
		_paramMVP.init(_shader, "MVP");
		_paramLimit.init(_shader, "alpha_limit");
		_paramStage.init(_shader, "stage");

		glCreateTextures(GL_TEXTURE_2D, 1, &idTexture);
		glTextureParameteri(idTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(idTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glCreateTextures(GL_TEXTURE_2D, 1, &colorTexture);
		glTextureParameteri(colorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(colorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glCreateFramebuffers(1, &fbo);
		glCreateRenderbuffers(1, &depthBuffer);

		makeFBO(800, 800);

		clearProg = glCreateProgram();
		const char* clearShaderSrc = R"(
			#version 430

			layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

			layout(std430, binding = 0) buffer IntArray {
				int arr[];
			};

			layout(location = 0) uniform int size;

			void main() {
				uint index = gl_GlobalInvocationID.x;
				if (index < size) {
					arr[index] = 0;
				}
			} 
			)";
		clearShader = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(clearShader, 1, &clearShaderSrc, nullptr);
		glAttachShader(clearProg, clearShader);
		glLinkProgram(clearProg);
	}

	void GaussianSurfaceRenderer::makeFBO(int w, int h)
	{
		resX = w;
		resY = h;

		glBindTexture(GL_TEXTURE_2D, idTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, resX, resY, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0);

		glBindTexture(GL_TEXTURE_2D, colorTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resX, resY, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		glNamedRenderbufferStorage(depthBuffer, GL_DEPTH_COMPONENT, resX, resY);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, idTexture, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	}

	int	GaussianSurfaceRenderer::process(int G, const GaussianData& mesh, const Camera& eye, IRenderTarget& target, float limit, sibr::Mesh::RenderMode mode, bool backFaceCulling)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		if (target.w() != resX || target.h() != resY)
		{
			makeFBO(target.w(), target.h());
		}

		// Solid pass
		GLuint drawBuffers[2];
		drawBuffers[0] = GL_COLOR_ATTACHMENT0;
		drawBuffers[1] = GL_COLOR_ATTACHMENT1;
		glDrawBuffers(2, drawBuffers);

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		_paramCamPos.set(eye.position());
		_paramLimit.set(limit);
		_paramStage.set(0);
		mesh.render(G);

		// Simple additive blendnig (no order)
		glDrawBuffers(1, drawBuffers);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		_paramStage.set(1);
		mesh.render(G);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);

		_shader.end();

		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glBlitNamedFramebuffer(
			fbo, target.fbo(),
			0, 0, resX, resY,
			0, 0, resX, resY,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);

		return 0;
	}

} /*namespace sibr*/
#pragma endregion ReDefine


// Define the types and sizes that make up the contents of each Gaussian 
 // in the trained model.
typedef sibr::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D + 1) * (D + 1) * 3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	std::vector<sibr::Vector3f>& normals,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
	{
		if (buff.compare("end_header") == 0)
			break;		
	}

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	normals.resize(count);
	shs.resize(count);
	opacities.resize(count);
	scales.resize(count);
	rot.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
		};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;
		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
		{
			scales[k].scale[j] = exp(points[i].scale.scale[j]);
		}

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
		normals[k] = sibr::Vector3f(points[i].n[0], points[i].n[1], points[i].n[2]);
		normals[k].normalize();
	}
	return count;
}

void savePly(const char* filename,
	const std::vector<Pos>& pos,
	const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<Scale>& scales,
	const std::vector<Rot>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" };
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S);

namespace sibr
{	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader;
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};


	sibr::DF_L::GaussianView::GaussianView(const sibr::DF_L::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device) :
		_scene(ibrScene),
		_dontshow(messageRead),
		_sh_degree(sh_degree),
		sibr::ViewBase(render_w, render_h)
	{
		int num_devices;
		CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
		_device = device;
		if (device >= num_devices)
		{
			if (num_devices == 0)
				SIBR_ERR << "No CUDA devices detected!";
			else
				SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
		}
		CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
		cudaDeviceProp prop;
		CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
		if (prop.major < 7)
		{
			SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
		}

		_pointbasedrenderer.reset(new PointBasedRenderer());
		_copyRenderer = new BufferCopyRenderer();
		_copyRenderer->flip() = true;
		_copyRenderer->width() = render_w;
		_copyRenderer->height() = render_h;

		std::vector<uint> imgs_ulr;
		const auto& cams = ibrScene->cameras()->inputCameras();
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			if (cams[cid]->isActive()) {
				imgs_ulr.push_back(uint(cid));
			}
		}
		_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

		// Load the PLY data (AoS) to the GPU (SoA)
		std::vector<Pos> pos;
		std::vector<Rot> rot;
		std::vector<Scale> scale;
		std::vector<float> opacity;
		std::vector<SHs<3>> shs;
		std::vector<sibr::Vector3f> normal;
		if (sh_degree == 0)
		{
			count = loadPly<0>(file, pos, shs, opacity, scale, rot, normal, _scenemin, _scenemax);
		}
		else if (sh_degree == 1)
		{
			count = loadPly<1>(file, pos, shs, opacity, scale, rot, normal, _scenemin, _scenemax);
		}
		else if (sh_degree == 2)
		{
			count = loadPly<2>(file, pos, shs, opacity, scale, rot, normal, _scenemin, _scenemax);
		}
		else if (sh_degree == 3)
		{
			count = loadPly<3>(file, pos, shs, opacity, scale, rot, normal, _scenemin, _scenemax);
		}

		_boxmin = _scenemin;
		_boxmax = _scenemax;

		int P = count;

		// Allocate and fill the GPU data
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&normal_cuda, sizeof(sibr::Vector3f) * P));
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(normal_cuda, normal.data(), sizeof(sibr::Vector3f) * P, cudaMemcpyHostToDevice));

		// Out
		int numPixels = _copyRenderer->width() * _copyRenderer->height();
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pixelNormals_cuda, sizeof(sibr::Vector3f) * numPixels));

		// Create space for view parameters
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

		float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

		gData = new GaussianData(P,
			(float*)pos.data(),
			(float*)rot.data(),
			(float*)scale.data(),
			opacity.data(),
			(float*)shs.data());

		_gaussianRenderer = new GaussianSurfaceRenderer();

		// Create GL buffer ready for CUDA/GL interop
		glCreateBuffers(1, &imageBuffer);
		glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

		if (useInterop)
		{
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
			}
			cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
			useInterop &= (cudaGetLastError() == cudaSuccess);
		}
		if (!useInterop)
		{
			fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
			cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
			_interop_failed = true;
		}

		geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
		binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
		imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

		// Set Directional Light
		lightDesc.dir_Dir = sibr::Vector3f(1, -2, 1);
		lightDesc.dir_Dir.normalize();
		lightDesc.dir_diffuse = sibr::Vector4f(1.f, 0.957f, 0.839f, 1.f);
		lightDesc.dir_diffusePower = 1.f;
		lightDesc.ambient = sibr::Vector4f(1.f, 1.f, 1.f, 1.f);
		lightDesc.ambientPower = 0.05f;
	}

	void sibr::DF_L::GaussianView::setScene(const sibr::DF_L::BasicIBRScene::Ptr& newScene)
	{
		_scene = newScene;

		// Tell the scene we are a priori using all active cameras.
		std::vector<uint> imgs_ulr;
		const auto& cams = newScene->cameras()->inputCameras();
		for (size_t cid = 0; cid < cams.size(); ++cid) {
			if (cams[cid]->isActive()) {
				imgs_ulr.push_back(uint(cid));
			}
		}
		_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
	}

	void sibr::DF_L::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
	{
		if (currMode == "Ellipsoids")
		{
			_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
		}
		else if (currMode == "Initial Points")
		{
			_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
		}
		else
		{
			// Convert view and projection to target coordinate system
			auto view_mat = eye.view();
			auto proj_mat = eye.viewproj();
			view_mat.row(1) *= -1;
			view_mat.row(2) *= -1;
			proj_mat.row(1) *= -1;

			// Compute additional view parameters
			float tan_fovy = tan(eye.fovy() * 0.5f);
			float tan_fovx = tan_fovy * eye.aspect();

			// Copy frame-dependent data to GPU
			CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

			float* image_cuda = nullptr;
			


			if (!_interop_failed)
			{
				// Map OpenGL buffer resource for use with CUDA
				size_t bytes;
				CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
				CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
				mappedSize = bytes;
			}
			else
			{
				image_cuda = fallbackBufferCuda;
			}

			// Rasterize - Pass1
			int* rects = _fastCulling ? rect_cuda : nullptr;
			float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
			float* boxmax = _cropping ? (float*)&_boxmax : nullptr;

			/*CudaRasterizer::DF_L::Rasterizer::forward(
				geomBufferFunc,
				binningBufferFunc,
				imgBufferFunc,
				count, _sh_degree, 16,
				background_cuda,
				_resolution.x(), _resolution.y(),
				pos_cuda,
				shs_cuda,
				nullptr,
				opacity_cuda,
				scale_cuda,
				_scalingModifier,
				rot_cuda,
				nullptr,
				view_cuda,
				proj_cuda,
				cam_pos_cuda,
				tan_fovx,
				tan_fovy,
				false,
				image_cuda,
				nullptr,
				rects,
				boxmin,
				boxmax,
				normal_cuda,
				pixelNormals_cuda,
				lightDesc
			);*/

			//float* f = new float[bytes];
			if (!_interop_failed)
			{
				if (mappedSize > 0)
				{
					copiedOutColor = new float[mappedSize / sizeof(float)];
					CUDA_SAFE_CALL(cudaMemcpy(copiedOutColor, image_cuda, mappedSize, cudaMemcpyDeviceToHost));
					
				}

				// Unmap OpenGL resource for use with OpenGL
				CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
			}
			else
			{
				CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
				glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
			}
			// Copy image contents to framebuffer
			_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
		}

		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
	}

	void sibr::DF_L::GaussianView::onUpdate(Input& input)
	{
	}

	void sibr::DF_L::GaussianView::onGUI()
	{
		// Generate and update UI elements
		const std::string guiName = "3D Gaussians";
		if (ImGui::Begin(guiName.c_str()))
		{
			if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
			{
				if (ImGui::Selectable("Splats"))
					currMode = "Splats";
				if (ImGui::Selectable("Initial Points"))
					currMode = "Initial Points";
				if (ImGui::Selectable("Ellipsoids"))
					currMode = "Ellipsoids";
				ImGui::EndCombo();
			}
		}
		if (currMode == "Splats")
		{
			ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
		}
		ImGui::Checkbox("Fast culling", &_fastCulling);

		ImGui::Checkbox("Crop Box", &_cropping);
		if (_cropping)
		{
			ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
			ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
			ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
			ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
			ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
			ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
			ImGui::InputText("File", _buff, 512);
			if (ImGui::Button("Save"))
			{
				std::vector<Pos> pos(count);
				std::vector<Rot> rot(count);
				std::vector<float> opacity(count);
				std::vector<SHs<3>> shs(count);
				std::vector<Scale> scale(count);
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
				CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
				//CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), normal_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
				savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax);
			}
		}

		// Light Setting


		ImGui::End();

		if (!*_dontshow && !accepted && _interop_failed)
			ImGui::OpenPopup("Error Using Interop");

		if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
			ImGui::SetItemDefaultFocus();
			ImGui::SetWindowFontScale(2.0f);
			ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
				" It did NOT work for your current configuration.\n"\
				" For highest performance, OpenGL and CUDA must run on the same\n"\
				" GPU on an OS that supports interop.You can try to pass a\n"\
				" non-zero index via --device on a multi-GPU system, and/or try\n" \
				" attaching the monitors to the main CUDA card.\n"\
				" On a laptop with one integrated and one dedicated GPU, you can try\n"\
				" to set the preferred GPU via your operating system.\n\n"\
				" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

			ImGui::Separator();

			if (ImGui::Button("  OK  ")) {
				ImGui::CloseCurrentPopup();
				accepted = true;
			}
			ImGui::SameLine();
			ImGui::Checkbox("Don't show this message again", _dontshow);
			ImGui::EndPopup();
		}
	}

	sibr::DF_L::GaussianView::~GaussianView()
	{
		// Cleanup
		cudaFree(pixelNormals_cuda);
		cudaFree(normal_cuda);

		cudaFree(pos_cuda);
		cudaFree(rot_cuda);
		cudaFree(scale_cuda);
		cudaFree(opacity_cuda);
		cudaFree(shs_cuda);

		cudaFree(view_cuda);
		cudaFree(proj_cuda);
		cudaFree(cam_pos_cuda);
		cudaFree(background_cuda);
		cudaFree(rect_cuda);

		if (!_interop_failed)
		{
			cudaGraphicsUnregisterResource(imageBufferCuda);
		}
		else
		{
			cudaFree(fallbackBufferCuda);
		}
		glDeleteBuffers(1, &imageBuffer);

		if (geomPtr)
			cudaFree(geomPtr);
		if (binningPtr)
			cudaFree(binningPtr);
		if (imgPtr)
			cudaFree(imgPtr);

		delete _copyRenderer;
	}

}


std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
		};
	return lambda;
}