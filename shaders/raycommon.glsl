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

// Homogeneous medium parameters (matches GPUMediumParams)
layout(set = 4, binding = 0, std140) uniform Medium {
    vec4 sigma_a_step;
    vec4 sigma_s_maxT;
    vec4 g_emis_density_pad;
} uMedium;

#define U_SIGMA_A       (uMedium.sigma_a_step.xyz)
#define U_STEP          (uMedium.sigma_a_step.w)
#define U_SIGMA_S       (uMedium.sigma_s_maxT.xyz)
#define U_MAXT          (uMedium.sigma_s_maxT.w)
#define U_G             (uMedium.g_emis_density_pad.x)
#define U_EMISSION      (uMedium.g_emis_density_pad.y)
#define U_DENSITY_SCALE (uMedium.g_emis_density_pad.z)
#define U_FOG_ENV       (uMedium.g_emis_density_pad.w)

vec3 sigma_t() { return U_SIGMA_A + U_SIGMA_S; }

// Safe, per-channel divide: a/b with b>=0; returns 0 where b==0
vec3 safeDiv(vec3 a, vec3 b) {
    b = max(b, vec3(1e-8));
    return a / b;
}

vec2 randomVec2(vec2 seed) {
    return vec2(fract(sin(seed.x * 12.9898 + seed.y * 78.233) * 43758.5453),
                fract(sin(seed.y * 34.7892 + seed.x * 56.1234) * 12345.6789));
}