
layout(binding = 3) uniform sampler2D hdrCondDistTex;
layout(binding = 4) uniform sampler2D hdrMarginalDistTex;

layout(binding = 5) uniform sampler2D hdrTex;
layout(binding = 6) uniform sampler3D textureMapsArrayTex;
layout( std140,binding = 7) buffer BVH{
	ivec4 bvh_nodes[ ];
};

layout(std140, binding = 8) buffer BBoxMin{
	vec4 bboxMin[];
};
layout(std140, binding = 9) buffer BBoxMax{
	vec4 bboxMax[];
};
layout(std140, binding = 10) buffer vertexIndicesTex{
ivec4 vertexIndices[];
};

layout(std140, binding = 11) buffer verticesTex{
vec4 vertices[];
};
layout(std140, binding = 12) buffer normalsTex{
vec4 normals[];
};
layout(std140, binding = 13) buffer materialsTex{
Material materials[];
};
layout(std140, binding = 14) buffer transformsTex{
mat4 transforms[];
};
layout(std140, binding = 15) buffer lightsTex{
Light lights[];
};



