#include "RenderingModeCustomized.hpp"

#include "core/graphics/RenderUtility.hpp"
#include "core/view/RenderingMode.hpp"
#include "core/assets/Resources.hpp"
#include "core/graphics/Image.hpp"

sibr::MonoRdrModeCustomized::MonoRdrModeCustomized()
{
	_clear = true;
	_quadShader.init("Texture",
		sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.vp")),
		sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.fp")));
}

void sibr::MonoRdrModeCustomized::render(ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport,
	IRenderTarget* optDest)
{
	/// TODO: clean everything. Resolution handling.

		//int w = (int)viewport.finalWidth();
		//int h = (int)viewport.finalHeight();

		//if (!_destRT || _destRT->w() != w || _destRT->h() != h)
		//	_destRT.reset( new RenderTarget(w, h) );
		//
		//view.onRenderIBR(*_destRT, eye);
		//_destRT->unbind();

		//_quadShader.begin();
		////if(_ibr->isPortraitAcquisition() && !_ibr->args().fullscreen)
		////	glViewport(0,0, _h, _w);
		////else
		////	glViewport(0,0, _w * _ibr->args().rt_factor, (_ibr->args().fullscreen ? screenHeight : _h) * _ibr->args().rt_factor);
		//viewport.use();
		////glViewport(0,0, size().x(), size().y());

		//glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _destRT->texture());
		//RenderUtility::renderScreenQuad(false /*_ibr->isPortraitAcquisition()*/);
		//_quadShader.end();

	int w = (int)viewport.finalWidth();
	int h = (int)viewport.finalHeight();

	if (!_destRT)// || _destRT->w() != w || _destRT->h() != h)
		_destRT.reset(new RenderTarget(w, h, SIBR_GPU_LINEAR_SAMPLING));
	glViewport(0, 0, w, h);
	_destRT->bind();

	if (_clear) {
		viewport.clear();
		// blend with previous
		view.preRender(*_destRT);
	}
	else {
		// can come from somewhere else
		view.preRender(*_prevR);
	}

	view.onRenderIBR(*_destRT, eye);
	_destRT->unbind();

	//show(*_destRT, "before");

	//glEnable (GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	//glDepthMask(GL_FALSE);

	//glEnable (GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	_quadShader.begin();
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, _destRT->texture());

	if (optDest) // Optionally you can render to another RenderTarget
	{
		glViewport(0, 0, optDest->w(), optDest->h());
		optDest->bind();
	}
	else
	{
		viewport.bind();
	}

	RenderUtility::renderScreenQuad(/*_ibr->isPortraitAcquisition()*/);

	if (optDest) // Optionally you can render to another RenderTarget
		optDest->unbind();

	_quadShader.end();

#if 0
	std::cerr << "End of render pass 1" << std::endl;
	show(*(_destRT));
#endif
}

void sibr::MonoRdrModeCustomized::Init_RT(const sibr::Viewport& viewport, bool bCreateDepthTexture)
{
	int w = (int)viewport.finalWidth();
	int h = (int)viewport.finalHeight();

	if (_destRT)// || _destRT->w() != w || _destRT->h() != h)
	{
		SIBR_ERR;
		return;
	}

	_destRT.reset(new RenderTarget(w, h, SIBR_GPU_LINEAR_SAMPLING));

	if (bCreateDepthTexture)
	{
		// 이 fbo에 대해
		//glBindFramebuffer(GL_FRAMEBUFFER, _destRT->fbo());
		_destRT->bind();
		glGenTextures(1, &depthTexture);
		glBindTexture(GL_TEXTURE_2D, depthTexture);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

		// 기존 Core코드에서 바인딩된 Depth RenderBuffer 해제
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
		// 해제된 DepthRenderBuffer 대신 DepthRenderTexture 바인딩
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

		////00
		/*glGenBuffers(1, &depthBuffer);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, depthBuffer);
		glBufferData(GL_PIXEL_PACK_BUFFER, w * h * sizeof(float), NULL, GL_STATIC_READ);
		glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, 0);*/
		
		

		_destRT->unbind();

		//glBindFramebuffer(GL_FRAMEBUFFER, 0);
		CHECK_GL_ERROR;
	}
}
