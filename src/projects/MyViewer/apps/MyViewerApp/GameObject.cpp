#include "GameObject.h"

sibr::GameObject::GameObject()
{
}

sibr::GameObject::GameObject(Mesh::Ptr pInMesh)
{
	m_pMesh = pInMesh.get();
	pInMesh.reset();
}

sibr::GameObject::~GameObject()
{
	delete m_pMesh;
	m_pMesh = nullptr;
}

void sibr::GameObject::Move(int axis, float moveSpeed)
{
	switch (axis)
	{
	case 0: // x
		m_Pos.x() += moveSpeed;
		break;
	case 1: // y
		m_Pos.y() += moveSpeed;
		break;
	case 2: // z
		m_Pos.z() += moveSpeed;
		break;
	}

	
	
}
