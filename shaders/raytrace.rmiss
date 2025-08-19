#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#include "raycommon.glsl"
#define M_PI 3.1415926535897932384626433832795

// Push constant structure for the ray tracer
struct PushConstantRay
{
    vec4 clearColor;
	uint64_t lightAddress;
    uint numLights;
};

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(set = 0, binding = 2) uniform sampler2D envmap;

layout(push_constant) uniform _PushConstantRay
{
    PushConstantRay pcRay;
};

void main()
{
    // Direction to env
    vec3 V = normalize(gl_WorldRayDirectionEXT);

    // Env lookup first (so we can tint by transmittance)
    vec2 uv = vec2(0.5 + atan(V.z, V.x) / (2.0 * M_PI), 0.5 - asin(V.y) / M_PI);
    vec3 env = texture(envmap, uv).xyz;

    // Homogeneous Beer–Lambert over maxT
    vec3 sig_t = sigma_t();
    float T = U_MAXT; // march limit (world units)
    vec3 Tr = exp(-sig_t * T);

    // Constant emission integral over the segment: integ. of emission * Tr(t) dt
    // With constant emission, closed-form per-channel: E * (1 - exp(-sigma_t T)) / sigma_t
    vec3 E = vec3(U_EMISSION);
    vec3 L_emis = E * safeDiv((vec3(1.0) - Tr), sig_t);

    // Fogged environment
    prd.hitValue = Tr * env + L_emis;
}