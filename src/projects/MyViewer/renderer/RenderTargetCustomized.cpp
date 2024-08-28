#include "RenderTargetCustomized.hpp"

namespace sibr
{
	// --- DEFINITIONS RenderTarget --------------------------------------------------

	RenderTargetRGBW::RenderTargetRGBW(void) {
		m_fbo = 0;
		m_depth_rb = 0;
		m_numtargets = 0;
		m_W = 0;
		m_H = 0;
	}

	RenderTargetRGBW::RenderTargetRGBW(uint w, uint h, uint flags, uint num) {
		RenderUtility::useDefaultVAO();

		m_W = w;
		m_H = h;

		bool alwaysFalse = false;

		int maxRenterTargets = 0;
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxRenterTargets);

		SIBR_ASSERT(num <= uint(maxRenterTargets) && num > 0);
		SIBR_ASSERT(!alwaysFalse || num == 1);

		if (flags & SIBR_GPU_INTEGER) {
			if (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format < 0) {
				throw std::runtime_error("Integer render  - format does not support integer mapping");
			}
		}

		glGenFramebuffers(1, &m_fbo);

		glGenRenderbuffers(1, &m_depth_rb); // depth buffer for color rt

		m_numtargets = num;
		m_autoMIPMAP = ((flags & SIBR_GPU_AUTOGEN_MIPMAP) != 0);

		m_msaa = ((flags & SIBR_GPU_MULSTISAMPLE) != 0);
		m_stencil = ((flags & SIBR_STENCIL_BUFFER) != 0);

		if (m_msaa && (m_numtargets != 1))
			throw std::runtime_error("Only one MSAA render target can be attached.");
		for (uint n = 0; n < m_numtargets; n++) {
			if (m_msaa)
				break;

			glGenTextures(1, &m_textures[n]);


			glBindTexture(GL_TEXTURE_2D, m_textures[n]);

			if (flags & SIBR_CLAMP_UVS) {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}

			/// \todo: following causes enum compare warning -Wenum-compare
			glTexImage2D(GL_TEXTURE_2D,
				0, 
				GL_RGBA32F,
				w, h,
				0,
				GL_RGBA,
				GL_FLOAT,
				NULL);
			//glTexImage2D(GL_TEXTURE_2D,
			//	0,
			//	(flags & SIBR_GPU_INTEGER)
			//	? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format
			//	: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format,
			//	w, h,
			//	0,
			//	(flags & SIBR_GPU_INTEGER)
			//	? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_format
			//	: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::format,
			//	GLType<typename PixelFormat::Type>::type,
			//	NULL);


			if (!m_autoMIPMAP) {
#if SIBR_COMPILE_FORCE_SAMPLING_LINEAR
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
				if (flags & SIBR_GPU_LINEAR_SAMPLING) {
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				}
				else {
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				}
#endif
			}
			else { /// \todo TODO: this crashes with 16F RT
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			}
		}


		if (!m_msaa) {
			if (!alwaysFalse) {
				glBindRenderbuffer(GL_RENDERBUFFER, m_depth_rb);
				if (!m_stencil)
					glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
				else
					glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);

				//CHECK_GL_ERROR;
				//glBindRenderbuffer(GL_RENDERBUFFER, m_stencil_rb);
				//glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, w, h);
				CHECK_GL_ERROR;
				glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
				for (uint n = 0; n < m_numtargets; n++) {
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + n, GL_TEXTURE_2D, m_textures[n], 0);
				}
				CHECK_GL_ERROR;
				if (!m_stencil)
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
				else
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
				//CHECK_GL_ERROR;
				//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_stencil_rb);
			}
			else {
				glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_textures[0], 0);
				glDrawBuffer(GL_NONE);
				glReadBuffer(GL_NONE);
			}
		}

		if (m_msaa) {
			uint msaa_samples = ((flags >> 7) & 0xF) << 2;

			if (msaa_samples == 0)
				throw std::runtime_error("Number of MSAA Samples not set. Please use SIBR_MSAA4X, SIBR_MSAA8X, SIBR_MSAA16X or SIBR_MSAA32X as an additional flag.");

			glGenTextures(1, &m_textures[0]);
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_textures[0]);
			CHECK_GL_ERROR;
			/// TODO: following causes enum compare warning -Wenum-compare
			glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE,
				msaa_samples,
				(flags & SIBR_GPU_INTEGER)
				? GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::int_internal_format
				: GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::internal_format,
				w, h,
				GL_TRUE
			);
			glBindRenderbuffer(GL_RENDERBUFFER, m_depth_rb);
			glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa_samples, GL_DEPTH_COMPONENT32, w, h);
			glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
			glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_textures[0], 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_rb);
		}

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			switch (status) {
			case GL_FRAMEBUFFER_UNSUPPORTED:
				throw std::runtime_error("Cannot create FBO - GL_FRAMEBUFFER_UNSUPPORTED error");
				break;
			default:
				SIBR_DEBUG(status);
				throw std::runtime_error("Cannot create FBO (unknow reason)");
				break;
			}
		}

		if (m_autoMIPMAP) {
			for (uint i = 0; i < m_numtargets; i++) {
				glBindTexture(GL_TEXTURE_2D, m_textures[i]);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		CHECK_GL_ERROR;
	}

	RenderTargetRGBW::~RenderTargetRGBW(void) {
		for (uint i = 0; i < m_numtargets; i++)
			glDeleteTextures(1, &m_textures[i]);
		glDeleteFramebuffers(1, &m_fbo);
		glDeleteRenderbuffers(1, &m_depth_rb);
		CHECK_GL_ERROR;
	}

	GLuint RenderTargetRGBW::depthRB() const {
		return m_depth_rb;
	}

	GLuint RenderTargetRGBW::texture(uint t) const {
		SIBR_ASSERT(t < m_numtargets);
		return m_textures[t];
	}

	GLuint RenderTargetRGBW::handle(uint t) const {
		SIBR_ASSERT(t < m_numtargets);
		return m_textures[t];
	}

	void RenderTargetRGBW::bind(void) {
		glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
		bool is_depth = (GLFormat<typename PixelFormat::Type, PixelFormat::NumComp>::isdepth != 0);
		if (!is_depth) {
			if (m_numtargets > 0) {
				GLenum drawbuffers[SIBR_MAX_SHADER_ATTACHMENTS];
				for (uint i = 0; i < SIBR_MAX_SHADER_ATTACHMENTS; i++)
					drawbuffers[i] = GL_COLOR_ATTACHMENT0 + i;
				glDrawBuffers(m_numtargets, drawbuffers);
			}
		}
		else {
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		}
	}

	void RenderTargetRGBW::unbind(void) {
		if (m_autoMIPMAP) {
			for (uint i = 0; i < m_numtargets; i++) {
				glBindTexture(GL_TEXTURE_2D, m_textures[i]);
				glGenerateMipmap(GL_TEXTURE_2D);
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void RenderTargetRGBW::clear(void) {
		clear(PixelFormat());
	}

	void RenderTargetRGBW::clear(const typename RenderTargetRGBW::PixelFormat& v) {
		bind();
		if (PixelFormat::NumComp == 1) {
			glClearColor(GLclampf(v[0]), 0, 0, 0);
		}
		else if (PixelFormat::NumComp == 2) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), 0, 0);
		}
		else if (PixelFormat::NumComp == 3) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), GLclampf(v[2]), 0);
		}
		else if (PixelFormat::NumComp == 4) {
			glClearColor(GLclampf(v[0]), GLclampf(v[1]), GLclampf(v[2]), GLclampf(v[3]));
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		unbind();
	}

	void RenderTargetRGBW::clearStencil() {
		bind();
		glClearStencil(0);
		glClear(GL_STENCIL_BUFFER_BIT);
		unbind();
	}

	void RenderTargetRGBW::clearDepth() {
		bind();
		glClear(GL_DEPTH_BUFFER_BIT);
		unbind();
	}

	uint   RenderTargetRGBW::numTargets(void)  const { return m_numtargets; }
	uint   RenderTargetRGBW::w(void)  const { return m_W; }
	uint   RenderTargetRGBW::h(void)  const { return m_H; }
	uint   RenderTargetRGBW::fbo(void)  const { return m_fbo; }
}