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
	mat4 invView;
	mat4 invProj;
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

// PCG integer hash. Preferred over the usual fract(sin(dot(...)) * large)
// trick, which relies on float precision loss for its randomness and so
// degenerates into visible banding once the seed grows large — exactly what
// happens when a frame counter is folded in.
uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uses the mantissa bits directly, giving a uniform value in [0,1).
float uint_to_unit_float(uint x) {
    return uintBitsToFloat(0x3f800000u | (x >> 9u)) - 1.0;
}

// Seeded from integers — pixel, frame and a per-sample index — rather than from
// anything float-derived. A seed taken from the ray direction reshuffles the
// entire noise pattern whenever ray setup changes in its last few bits, which
// makes otherwise-identical images compare as different.
vec2 randomVec2(uvec2 pixel, uint frame, uint index) {
    uint h = pcg_hash(pixel.x + pcg_hash(pixel.y + pcg_hash(frame + pcg_hash(index))));
    return vec2(uint_to_unit_float(h), uint_to_unit_float(pcg_hash(h)));
}