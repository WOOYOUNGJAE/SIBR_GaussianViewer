#pragma once

# include <core/graphics/Shader.hpp>
# include <core/graphics/Texture.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/graphics/Camera.hpp>

# include <core/renderer/Config.hpp>

// Simple Mesh Rasterization Renderer for practice
class MeshRenderer
{
//public:
//	typedef std::shared_ptr<MeshRenderer>	Ptr;
//
//public:
//
//	/// Constructor.
//	MeshRenderer(void);
//
//	/** Render the mesh using its vertices colors, interpolated over triangles.
//	\param mesh the mesh to render
//	\param eye the viewpoint to use
//	\param dst the destination rendertarget
//	\param mode the rendering mode of the mesh
//	\param backFaceCulling should backface culling be performed
//	*/
//	int	process(
//		int G,
//		/*input*/	const GaussianData& mesh,
//		/*input*/	const Camera& eye,
//		/*output*/	IRenderTarget& dst,
//		float alphaLimit,
//		/*mode*/    sibr::Mesh::RenderMode mode = sibr::Mesh::FillRenderMode,
//		/*BFC*/     bool backFaceCulling = true);
//
//	void makeFBO(int w, int h);
//
//private:
//
//	GLuint idTexture;
//	GLuint colorTexture;
//	GLuint depthBuffer;
//	GLuint fbo;
//	int resX, resY;
//
//	GLShader			_shader; ///< Color shader.
//	GLParameter			_paramMVP; ///< MVP uniform.
//	GLParameter			_paramCamPos;
//	GLParameter			_paramLimit;
//	GLParameter			_paramStage;
//	GLuint clearProg;
//	GLuint clearShader;
};

