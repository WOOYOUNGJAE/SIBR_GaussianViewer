#pragma once
#include "core/view/RenderingMode.hpp"
#include "Config.hpp"

namespace sibr
{
	class RenderTargetRGBW;
	class SIBR_EXP_ULR_EXPORT MonoRdrModeCustomized : public IRenderingMode
	{
	public:

		/// Constructor.
		MonoRdrModeCustomized(void);

		/** Perform rendering of a view.
		 *\param view the view to render
		 *\param eye the current camera
		 *\param viewport the current viewport
		 *\param optDest an optional destination RT
		 */
		void	render(ViewBase& view, const sibr::Camera& eye, const sibr::Viewport& viewport, IRenderTarget* optDest = nullptr);

		/** Get the current rendered image as a CPU image
		 *\param current_img will contain the content of the RT */
		void destRT2img(sibr::ImageRGB& current_img)
		{
			_destRT->readBack(current_img);
			return;
		}

		/** \return the common RT. */
		virtual const std::unique_ptr<RenderTargetRGB>& lRT() { return _destRT; }
		/** \return the common RT. */
		virtual const std::unique_ptr<RenderTargetRGB>& rRT() { return _destRT; }

		void Init_RT(const sibr::Viewport& viewport);
	private:
		sibr::GLShader							_quadShader; ///< Passthrough shader.
		std::unique_ptr<RenderTarget>		_destRT; ///< Common destination RT.
	};

}