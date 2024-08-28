#pragma once
#include "core/graphics/RenderTarget.hpp"
#include "Config.hpp"

namespace sibr
{
	// RT with Color Texture and Depth Texture
	class SIBR_EXP_ULR_EXPORT RenderTargetRGBW : public sibr::IRenderTarget
	{
		SIBR_DISALLOW_COPY(RenderTargetRGBW);
	public:
		//typedef		Image<float, 4>		PixelImage;
		typedef		Image<unsigned char, 3>		PixelImage;
		typedef		typename PixelImage::Pixel		PixelFormat;
		typedef		std::shared_ptr<RenderTargetRGBW>	Ptr;
		typedef		std::unique_ptr<RenderTargetRGBW>	UPtr;

	private:

		GLuint m_fbo = 0; ///< Framebuffer handle.
		GLuint m_depth_rb = 0; ///< Depth renderbuffer handle.
		GLuint m_stencil_rb = 0; ///< Stencil renderbuffer handle.
		GLuint m_textures[SIBR_MAX_SHADER_ATTACHMENTS]; ///< Color texture handles.
		uint   m_numtargets = 0; ///< Number of active color attachments.
		bool   m_autoMIPMAP = false; ///< Generate mipmaps on the fly.
		bool   m_msaa = false; ///< Use multisampled targets.
		bool   m_stencil = false; ///< Has a stencil buffer.
		uint   m_W = 0; ///< Width.
		uint   m_H = 0; ///< Height.

	public:

		/// Constructor.
		RenderTargetRGBW(void);

		/** Constructor and allocation.
		\param w the target width
		\param h the target height
		\param flags options
		\param num the number of color attachments.
		*/
		RenderTargetRGBW(uint w, uint h, uint flags = 0, uint num = 1);

		/// Destructor.
		~RenderTargetRGBW(void);

		/** Get the texture handle of the t-th color attachment.
		\param t the color attachment slot
		\return the texture handle
		\deprecated Use handle instead.
		*/
		GLuint texture(uint t = 0) const;

		/** Get the texture handle of the t-th color attachment.
		\param t the color attachment slot
		\return the texture handle
		*/
		GLuint handle(uint t = 0) const;

		/** \return the depth buffer handle. */
		GLuint depthRB() const;

		/** Bind the rendertarget for drawing. All color buffers are bound, along
			with the depth and optional stencil buffers.*/
		void bind(void);

		/** Unbind the rendertarget.
		\note This will bind the window rendertarget. */
		void unbind(void);

		/** Clear the rendertarget buffers with default values.
		 * \warning This function will unbind the render target after clearing.
		 */
		void clear(void);

		/** Clear the rendertarget buffers, using a custom clear color.
		 * \param v the clear color
		 * \warning This function will unbind the render target after clearing.
		 * \bug This function does not rescale values for uchar (so background is either 0 or 1)
		 */
		void clear(const typename RenderTargetRGBW::PixelFormat& v);

		/** Clear the stencil buffer only. */
		void clearStencil(void);

		/** Clear the depth buffer only. */
		void clearDepth(void);

		/** Readback the content of a color attachment into an sibr::Image on the CPU.
		\param image will contain the texture content
		\param target the color attachment index to read
		\warning Might cause a GPU flush/sync.
		*/
		template <typename TType, uint NNumComp>
		void readBack(sibr::Image<TType, NNumComp>& image, uint target = 0) const;

		/** Readback the content of a color attachment into a cv::Mat on the CPU.
		\param image will contain the texture content
		\param target the color attachment index to read
		\warning Might cause a GPU flush/sync.
		*/
		template <typename TType, uint NNumComp>
		void readBackToCVmat(cv::Mat& image, uint target = 0) const;

		/** Readback the content of the depth attachment into an sibr::Image on the CPU.
		\param image will contain the depth content
		\warning Might cause a GPU flush/sync.
		\warning Image orientation might be inconsistent with readBack (flip around horizontal axis).
		*/
		template <typename TType, uint NNumComp>
		void readBackDepth(sibr::Image<TType, NNumComp>& image) const;

		/** \return the number of active color targets. */
		uint   numTargets(void)  const;

		/** \return the target width. */
		uint   w(void)  const;

		/** \return the target height. */
		uint   h(void)  const;

		/** \return the framebuffer handle. */
		GLuint fbo(void)  const;
	};


	template <typename T_IType, uint N_INumComp>
	void RenderTargetRGBW::readBack(sibr::Image<T_IType, N_INumComp>& img, uint target) const {
		//void RenderTargetRGBW::readBack(PixelImage& img, uint target) const {
		glFinish();
		if (target >= m_numtargets)
			SIBR_ERR << "Reading back texture out of bounds" << std::endl;

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				PixelImage buffer(m_W, m_H);

				GLenum drawbuffers = GL_COLOR_ATTACHMENT0 + target;
				glDrawBuffers(1, &drawbuffers);
				glReadBuffer(drawbuffers);

				glReadPixels(0, 0, m_W, m_H,
					GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
					GLType<typename PixelFormat::Type>::type,
					buffer.data()
				);

				sibr::Image<T_IType, N_INumComp>	out;
				img.fromOpenCV(buffer.toOpenCV());
			}
		}
		else
			SIBR_ERR << "RenderTarget::readBack: This function should be specialized "
			"for handling depth buffer." << std::endl;
		img.flipH();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

	}


	template <typename T_IType, uint N_INumComp>
	void RenderTargetRGBW::readBackToCVmat(cv::Mat& img, uint target) const {

		using Infos = GLTexFormat<cv::Mat, T_IType, N_INumComp>;

		if (target >= m_numtargets)
			SIBR_ERR << "Reading back texture out of bounds" << std::endl;

		cv::Mat tmp(m_H, m_W, Infos::cv_type());

		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (Infos::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				GLenum drawbuffers = GL_COLOR_ATTACHMENT0 + target;
				glDrawBuffers(1, &drawbuffers);
				glReadBuffer(drawbuffers);

				glReadPixels(0, 0, m_W, m_H,
					Infos::format,
					Infos::type,
					Infos::data(tmp)
				);
			}
		}
		else {
			SIBR_ERR << "RenderTarget::readBack: This function should be specialized "
				"for handling depth buffer." << std::endl; \
		}
		img = Infos::flip(tmp);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	template <typename T_IType, uint N_INumComp>
	void RenderTargetRGBW::readBackDepth(sibr::Image<T_IType, N_INumComp>& image) const {
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

		glReadBuffer(GL_COLOR_ATTACHMENT0);

		sibr::Image<float, 1> buffer(m_W, m_H);
		glReadPixels(0, 0, m_W, m_H,
			GL_DEPTH_COMPONENT,
			GL_FLOAT,
			buffer.data()
		);

		sibr::Image<T_IType, N_INumComp>	out(buffer.w(), buffer.h());
		for (uint y = 0; y < buffer.h(); ++y)
			for (uint x = 0; x < buffer.w(); ++x)
				out.color(x, y, sibr::ColorRGBA(1, 1, 1, 1.f) * buffer(x, y)[0]);
		image = std::move(out);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

}