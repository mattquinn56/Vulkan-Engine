struct RenderLight {
    vec4 position; // a is intensity
    vec4 color; // a is type, 0 is point, 1 is ambient
	// 3 vertex positions and a normal, for an area light
	vec4 v0;
	vec4 v1;
	vec4 v2;
	vec4 normal;
};

layout(set = 1, binding = 0) uniform SceneData {   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
    vec4 numLights; // x is num lights
    RenderLight lights[16];
} sceneData;

struct hitPayload
{
    vec3 hitValue;
    int recursionDepth;
};

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
};

struct MaterialRT
{
    vec4 colorFactors;
    vec4 metal_rough_factors; // x is reflectivity proportion (metal), y is specular intensity proportion (roughness)
	int textureID;
};

// Push constant structure for the ray tracer
struct PushConstantRay
{
    vec4 clearColor;
};