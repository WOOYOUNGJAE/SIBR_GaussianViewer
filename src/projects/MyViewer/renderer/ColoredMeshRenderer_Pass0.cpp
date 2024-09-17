
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

		/*glCreateTextures(GL_TEXTURE_2D, 1, &idTexture);
		glTextureParameteri(idTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(idTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);*/

		// Color Texture
		glCreateTextures(GL_TEXTURE_2D, 1, &colorTexture);
		glTextureParameteri(colorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(colorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		// Depth Texture
		glCreateTextures(GL_TEXTURE_2D, 1, &depthTexture);
		glTextureParameteri(depthTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(depthTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


		glCreateFramebuffers(1, &fbo);

		//glCreateRenderbuffers(1, &depthBuffer);


		makeFBO(800, 800);
	}

	void ColoredMeshRenderer_Pass0::process(const Mesh& mesh, const Camera& eye, IRenderTarget& target, sibr::Mesh::RenderMode mode, bool backFaceCulling)
	{
		////glViewport(0.f, 0.f, target.w(), target.h());
		//target.bind();
		//glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		//_shader.begin();
		//_paramMVP.set(eye.viewproj());
		//mesh.render(true, backFaceCulling);
		//_shader.end();
		//target.unbind();

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		CHECK_GL_ERROR;

		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		CHECK_GL_ERROR;

		if (target.w() != resX || target.h() != resY)
		{
			makeFBO(target.w(), target.h());
		}

		// Solid pass
		GLuint drawBuffers[1];
		drawBuffers[0] = GL_COLOR_ATTACHMENT0; // color 
		//drawBuffers[1] = GL_COLOR_ATTACHMENT1; // ID
		glDrawBuffers(1, drawBuffers);
		CHECK_GL_ERROR;
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
		_shader.begin();
		_paramMVP.set(eye.viewproj());
		mesh.render();

		_shader.end();

		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glBlitNamedFramebuffer(
			fbo, target.fbo(),
			0, 0, resX, resY,
			0, 0, resX, resY,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);

	}

	void ColoredMeshRenderer_Pass0::makeFBO(int w, int h)
	{
		resX = w;
		resY = h;

		glBindTexture(GL_TEXTURE_2D, colorTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resX, resY, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindTexture(GL_TEXTURE_2D, depthTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, resX, resY, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		//glNamedRenderbufferStorage(depthBuffer, GL_DEPTH_COMPONENT, resX, resY);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
		//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);


	}
} /*namespace sibr*/
