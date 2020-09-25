struct Camera
{
	vec3 pos;
	vec3 lookat;
	float fov;

};
layout(binding = 2) uniform UBO
{			vec3 randomVector;
			float aspectRatio;

			vec2 screenResolution;
			float hdrTexSize;
			float hdrResolution;

			int numOfLights;
			int maxDepth;
			int topBVHIndex;
			int vertIndicesSize;
			int numTilesX, numTilesY, tileX, tileY;
			Camera camera;
		

} ubo;

int maxDepth = 2;
float hdrMultiplier = 0.0;
bool useEnvMap = false;
#define PI        3.14159265358979323
#define TWO_PI    6.28318530717958648
#define INFINITY  1000000.0
#define EPS 0.001

// Global variables

mat4 transform;

vec2 seed;
vec3 tempTexCoords;
struct Ray { vec3 origin; vec3 direction; };
struct Material { vec4 albedo; vec4 emission; vec4 param; vec4 texIDs; };
struct Light { vec3 position; vec3 emission; vec3 u; vec3 v; vec3 radiusAreaType; };
struct State { vec3 normal; vec3 ffnormal; vec3 fhp; bool isEmitter; int depth; float hitDist; vec2 texCoord; vec3 bary; ivec3 triID; int matID; Material mat; bool specularBounce; };
struct BsdfSampleRec { vec3 bsdfDir; float pdf; };
struct LightSampleRec { vec3 surfacePos; vec3 normal; vec3 emission; float pdf; };
struct Node
{
	int leftIndex;
	int rightIndex;
	int leaf;
};
struct Indices
{
	int x, y, z;
};



//-----------------------------------------------------------------------
float rand()
//-----------------------------------------------------------------------
{
	seed -= vec2(ubo.randomVector.x * ubo.randomVector.y);
	return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}