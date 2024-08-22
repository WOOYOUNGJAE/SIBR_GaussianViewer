#ifndef MY_STRUCT
#define MY_STRUCT

#include "core/system/Vector.hpp"

struct LIGHT_DESC
{
// common
	sibr::Vector4f ambient; // Environment Light
	float ambientPower; // 0 ~ 1
// Directional Light
	sibr::Vector3f dir_Dir; // Directional Light Dir
	sibr::Vector4f dir_diffuse; // DirLight Color
	float dir_diffusePower; // 0 ~ 1
// TODO: Point Light
};


#endif