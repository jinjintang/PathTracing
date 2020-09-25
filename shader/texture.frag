#version 450
layout(binding = 0) uniform UBO
{         int first;

} ubo;


layout (binding = 1) uniform sampler2D samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;
vec4 ToneMap(in vec4 c, float limit)
{
	float luminance = 0.3*c.x + 0.6*c.y + 0.1*c.z;

	return c * 1.0 / (1.0 + luminance / limit);
}

void main() 
{
   vec4 color = texture(samplerColor, vec2(inUV.s, 1.0 - inUV.t))/(1.0*ubo.first);
   outFragColor = pow(ToneMap(color, 1.5), vec4(1.0 / 2.2));
   
}