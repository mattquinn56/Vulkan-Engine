const int RLstride = 64; // total size of RenderLight, in bytes
struct RenderLight {
    vec4 position; // if directional light, this is direction. if area light, this is v2. alpha channel is intensity
    vec4 color; // alpha is type, 0 is point, 1 is ambient (no pos data used), 2 is directional (pos data is direction), 3 is area
	vec4 v0; // this and below is only populated if area light
	vec4 v1;
};

layout(set = 1, binding = 0) uniform SceneData {   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 data; // x is num frames, y is enable sampling
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

vec2 randomVec2(vec2 seed) {
    return vec2(fract(sin(seed.x * 12.9898 + seed.y * 78.233) * 43758.5453),
                fract(sin(seed.y * 34.7892 + seed.x * 56.1234) * 12345.6789));
}