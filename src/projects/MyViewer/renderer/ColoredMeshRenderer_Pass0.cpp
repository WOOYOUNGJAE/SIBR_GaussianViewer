
/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#include "ColoredMeshRenderer_Pass0.hpp"
#include "core/graphics/Texture.hpp"
namespace sibr {
	ColoredMeshRenderer_Pass0::ColoredMeshRenderer_Pass0(void)
	{
		_shader.init("ColoredMesh",
			sibr::loadFile(sibr::getShadersDirectory("core") + "/colored_mesh.vert"),
			sibr::loadFile(sibr::getShadersDirectory("core") + "/colored_mesh.frag"));
		_paramMVP.init(_shader, "MVP");
	}

	void	ColoredMeshRenderer_Pass0::process(const Mesh& mesh, const Camera& eye, IRenderTarget& target, sibr::Mesh::RenderMode mode, bool backFaceCulling)
	{
		//glViewport(0.f, 0.f, target.w(), target.h());
		target.bind();
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		mesh.render(true, backFaceCulling);
		_shader.end();
		target.unbind();
	}

} /*namespace sibr*/
