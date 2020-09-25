

#version 450
#extension GL_GOOGLE_include_directive : enable


layout (binding = 0) uniform sampler2D samplerColor;
layout(binding = 1,rgba32f) uniform   readonly image2D readImage;
#include "common/Globals.glsl"
#include "common/Uniforms.glsl"

#include "common/Intersection.glsl"
#include "common/Sampling.glsl"
#include "common/Anyhit.glsl"
#include "common/Closesthit.glsl"
#include "common/UE4BRDF.glsl"
#include "common/GlassBSDF.glsl"
#include "common/Pathtrace.glsl"

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragColor;


float map(float value, float low1, float high1, float low2, float high2)
{
	return low2 + ((value - low1) * (high2 - low2)) / (high1 - low1);
}
void main()
{
	float invTileWidth=1.0/ubo.numTilesX;
	float invTileHeight=1.0/ubo.numTilesY;
	float tileX=ubo.tileX;
	float tileY=ubo.tileY;
    seed = inUV;

	float xoffset = -1.0 + 2.0 * invTileWidth * float(tileX);
	float yoffset = -1.0 + 2.0 * invTileHeight * float(tileY);

	vec2 uv = inUV;
	uv.x = map(uv.x, 0.0, 1.0, xoffset, xoffset + 2.0 * invTileWidth);
	uv.y = map(uv.y, 0.0, 1.0, yoffset, yoffset + 2.0 * invTileHeight);

	vec3 rayO =ubo.camera.pos;
	vec3 rayD = normalize(vec3( uv * vec2(ubo.aspectRatio, 1.0), 1.0));
	Ray ray = Ray(rayO, rayD);

	vec2 coords;
	coords.x = map(inUV.x, 0.0, 1.0, invTileWidth * float(tileX), invTileWidth * float(tileX) + invTileWidth);
	coords.y = map(inUV.y, 0.0, 1.0, invTileHeight * float(tileY), invTileHeight * float(tileY) + invTileHeight);

	vec3 pixelColor=PathTrace(ray)+texture(samplerColor,coords).rgb;

	outFragColor=vec4(pixelColor.rgb ,1);
	

}