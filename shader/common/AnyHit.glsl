//-----------------------------------------------------------------------
bool AnyHit(Ray r, float maxDist)
//-----------------------------------------------------------------------
{
	int stack[64];
	int ptr = 0;
	stack[ptr++] = -1;

	int idx = ubo.topBVHIndex;
	float leftHit = 0.0;
	float rightHit = 0.0;

	int currMatID = 0;
	bool meshBVH = false;

	Ray r_trans;
	mat4 temp_transform;
	r_trans.origin = r.origin;
	r_trans.direction = r.direction;

	while (idx > -1 || meshBVH)
	{
		int n = idx;

		if (meshBVH && idx < 0)
		{
			meshBVH = false;

			idx = stack[--ptr];

			r_trans.origin = r.origin;
			r_trans.direction = r.direction;
			continue;
		}
		int index = n;


		int leftIndex = bvh_nodes[index].x;
		int rightIndex = bvh_nodes[index].y;
		int leaf = bvh_nodes[index].z;
	
		if (leaf > 0)
		{
			for (int i = 0; i < rightIndex; i++) // Loop through indices
			{
				ivec3 vert_indices = vertexIndices[index].xyz;

				vec4 v0 = vertices[vert_indices.x].xyzw;
				vec4 v1 = vertices[vert_indices.y].xyzw;
				vec4 v2 = vertices[vert_indices.z].xyzw;


				vec3 e0 = v1.xyz - v0.xyz;
				vec3 e1 = v2.xyz - v0.xyz;
				vec3 pv = cross(r_trans.direction, e1);
				float det = dot(e0, pv);

				vec3 tv = r_trans.origin - v0.xyz;
				vec3 qv = cross(tv, e0);

				vec4 uvt;
				uvt.x = dot(tv, pv);
				uvt.y = dot(r_trans.direction, qv);
				uvt.z = dot(e1, qv);
				uvt.xyz = uvt.xyz / det;
				uvt.w = 1.0 - uvt.x - uvt.y;

				if (all(greaterThanEqual(uvt, vec4(0.0))) && uvt.z < maxDist)
					return true;
			}
		}
		else if (leaf < 0)
		{
			idx = leftIndex;

			temp_transform = transforms[-leaf - 1];

			r_trans.origin = vec3(inverse(temp_transform) * vec4(r.origin, 1.0));
			r_trans.direction = vec3(inverse(temp_transform) * vec4(r.direction, 0.0));

			stack[ptr++] = -1;
			meshBVH = true;
			currMatID = rightIndex;
			continue;
		}
		else
		{

			leftHit = AABBIntersect(bboxMin[leftIndex].xyz, bboxMax[leftIndex].xyz, r_trans);
			rightHit = AABBIntersect(bboxMin[rightIndex].xyz, bboxMax[rightIndex].xyz, r_trans);
			if (leftHit > 0.0 && rightHit > 0.0)
			{
				int deferred = -1;
				if (leftHit > rightHit)
				{
					idx = rightIndex;
					deferred = leftIndex;
				}
				else
				{
					idx = leftIndex;
					deferred = rightIndex;
				}

				stack[ptr++] = deferred;
				continue;
			}
			else if (leftHit > 0.)
			{
				idx = leftIndex;
				continue;
			}
			else if (rightHit > 0.)
			{
				idx = rightIndex;
				continue;
			}
		}
		idx = stack[--ptr];
	}

	return false;
}