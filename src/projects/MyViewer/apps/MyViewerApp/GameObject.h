#pragma once
# include "core/graphics/Mesh.hpp"
# include "core/system/Transform3.hpp"

namespace sibr
{
	class GameObject
	{
	public:
		SIBR_CLASS_PTR(GameObject);
	public:
		GameObject();
		GameObject(Mesh::Ptr pInMesh);
		~GameObject();
	public:
		Mesh* Get_Mesh() { return m_pMesh; }
		Vector3f Position() { return m_Pos; }
		Vector3f& Position_Ref() { return m_Pos; }
		Matrix4f WorldMatrix() { return m_WorldMat; }
	public:
		void Move(int axis, float moveSpeed);

	private:
		Mesh* m_pMesh = nullptr; 
		Vector3f m_Pos{};
		Matrix4f m_WorldMat{};
	};

}