#pragma once
#include "core/renderer/ColoredMeshRenderer.hpp"
#include "Config.hpp"

namespace sibr {

	/** Render a mesh colored using the per-vertex color attribute.
	\ingroup sibr_renderer
	*/
	class SIBR_EXP_ULR_EXPORT ColoredMeshRenderer_Pass0
	{
	public:
		typedef std::shared_ptr<ColoredMeshRenderer_Pass0>	Ptr;

	public:

		/// Constructor.
		ColoredMeshRenderer_Pass0(void);

		/** Render the mesh using its vertices colors, interpolated over triangles.
		\param mesh the mesh to render
		\param eye the viewpoint to use
		\param dst the destination rendertarget
		\param mode the rendering mode of the mesh
		\param backFaceCulling should backface culling be performed
		*/
		void	process(
			/*input*/	const Mesh& mesh,
			/*input*/	const Camera& eye,
			/*output*/	IRenderTarget& dst,
			/*mode*/    sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
			/*BFC*/     bool backFaceCulling = true);

	private:

		GLShader			_shader; ///< Color shader.
		GLParameter			_paramMVP; ///< MVP uniform.

	};

} /*namespace sibr*/


