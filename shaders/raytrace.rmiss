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
    //prd.hitValue = pcRay.clearColor.xyz;

    vec3 normal = normalize(gl_WorldRayDirectionEXT);
    vec2 uv = vec2(0.5 + atan(normal.z, normal.x) / (2.0 * M_PI), 0.5 - asin(normal.y) / M_PI);

    //vec3 dir = gl_WorldRayDirectionEXT;
    //float azim = atan(dir.z, dir.x);
    //float elev = atan(dir.y, dir.x);
    //vec2 uv = vec2(.5) + vec2(azim / (2 * M_PI), elev / M_PI);
    vec4 envColor = texture(envmap, uv);
    prd.hitValue = envColor.xyz;
}