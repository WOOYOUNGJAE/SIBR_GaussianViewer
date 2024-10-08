#pragma once
#include "Scene.hpp"
#include "projects/MyViewer/renderer/GaussianView.hpp"
# include <../src/projects/MyViewer/renderer/ColoredMeshRenderer_Pass0.hpp>
#include "MyStructs.h"
// Views.hpp
namespace sibr
{
	class RenderTargetRGBW;
	namespace DF_L
	{
		/**
		 * \class RemotePointView
		 * \brief Wrap a ULR renderer with additional parameters and information.
		 */
		class GaussianView : public sibr::ViewBase
		{
			SIBR_CLASS_PTR(GaussianView);

		public:

			/**
			 * Constructor
			 * \param ibrScene The scene to use for rendering.
			 * \param render_w rendering width
			 * \param render_h rendering height
			 */
			GaussianView(const sibr::DF_L::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* message_read, int sh_degree, bool white_bg = false, bool useInterop = true, int device = 0);

			/** Replace the current scene.
			 *\param newScene the new scene to render */
			void setScene(const sibr::DF_L::BasicIBRScene::Ptr& newScene);

			/**
			 * Perform rendering. Called by the view manager or rendering mode.
			 * \param dst The destination rendertarget.
			 * \param eye The novel viewpoint.
			 */
			void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

			/**
			 * Update inputs (do nothing).
			 * \param input The inputs state.
			 */
			void onUpdate(Input& input) override;

			/**
			 * Update the GUI.
			 */
			void onGUI() override;

			/** \return a reference to the scene */
			const std::shared_ptr<sibr::DF_L::BasicIBRScene>& getScene() const { return _scene; }

			void Ready_MeshRendererOutputResource(GLuint texture, GLuint depthTexture);

			virtual ~GaussianView() override;

			bool* _dontshow;

		protected:

			std::string currMode = "Splats";

			bool _cropping = false;
			sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
			char _buff[512] = "cropped.ply";

			bool _fastCulling = true;
			int _device = 0;
			int _sh_degree = 3;

			int count;
			float* pos_cuda;
			float* rot_cuda;
			float* scale_cuda;
			float* opacity_cuda;
			float* shs_cuda;
			int* rect_cuda;
			float* meshRendererRGBArr_cuda = nullptr;
			float* meshRendererDepthArr_cuda = nullptr;

			GLuint imageBuffer;
			cudaGraphicsResource_t imageBufferCuda;
			cudaGraphicsResource_t meshRendererOutputImageCuda;
			cudaGraphicsResource_t meshRendererDepthCuda;
			GLuint depthBufferRef; // Only copy GLuint number that already Created
			cudaGraphicsResource_t depthBufferCuda;


			size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
			void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
			std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

			float* view_cuda;
			float* proj_cuda;
			float* cam_pos_cuda;
			float* background_cuda;

			float _scalingModifier = 1.0f;
			GaussianData* gData;

			bool _interop_failed = false;
			std::vector<char> fallback_bytes;
			float* fallbackBufferCuda = nullptr;
			bool accepted = false;


			std::shared_ptr<sibr::DF_L::BasicIBRScene> _scene; ///< The current scene.
			PointBasedRenderer::Ptr _pointbasedrenderer;
			BufferCopyRenderer* _copyRenderer;
			GaussianSurfaceRenderer* _gaussianRenderer;
			ColoredMeshRenderer_Pass0* m_coloredMeshRenderer = nullptr;

			size_t imageCudaMappedSize = 0;

		public:
			LIGHT_DESC lightDesc{};
		private: // Deferred Lighting
			float* normal_cuda = nullptr; // normals in
			float* pixelNormals_cuda = nullptr; // normals out
		public:
		};

	}
}

