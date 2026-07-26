#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#include "raycommon.glsl"

const int POINT = 0;
const int AMBIENT = 1;
const int DIRECTIONAL = 2;
const int AREA = 3;
const float EPSILON = .01;
const float T_MAX = 10000.0;
const int MAX_RECURSION = 4; // should be the same as MAX_RECURSION in vk_raytracer.h
const bool USE_METAL_ROUGH_TEX = false;

// Information of a obj model when referenced in a shader
struct ObjDesc {
    uint64_t vertexAddress; // Address of the vertex buffer
    uint64_t indexAddress; // Address of the index buffer
    uint64_t materialAddress; // Address of the material buffer
};

// Push constant structure for the ray tracer
struct PushConstantRay
{
    vec4 clearColor;
	uint64_t lightAddress;
    uint numLights;
    uint useMicrofacet; // 0=legacy, 1=GGX
};

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;
hitAttributeEXT vec2 attribs;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) buffer Indices { ivec3 i[]; };
layout(buffer_reference, scalar) buffer Material { MaterialRT m; };
layout(buffer_reference, scalar) buffer Light { RenderLight rl; };
layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 2, binding = 0, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;
//layout(set = 3, binding = 0, scalar) buffer ColImage2D { sampler2d i[]; };
//layout(set = 3, binding = 1, scalar) buffer MetalRoughImage2D { sampler2d i[]; };

const int TEX_MAX = 256;
layout(set = 3, binding = 0) uniform sampler2D ColImage2D[TEX_MAX];
layout(set = 3, binding = 1) uniform sampler2D MetalRoughImage2D[TEX_MAX];

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };


// Microfacet helpers (GGX + Schlick Fresnel + Smith)
const float PI = 3.14159265358979323846;

float saturate(float x) { return clamp(x, 0.0, 1.0); }

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    // UE4-style Schlick
    return F0 + (1.0 - F0) * pow(1.0 - saturate(cosTheta), 5.0);
}

float D_GGX(float NdotH, float alpha)
{
    float a2 = alpha * alpha;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-6);
}

float G_SchlickSmith(float NdotV, float NdotL, float alpha)
{
    // k from Epic (alpha remap)
    float k = (alpha + 1.0);
    k = (k * k) / 8.0;
    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    return gV * gL;
}

// Returns BRDF * NdotL (no light color or intensity applied)
vec3 brdf_ggx_smith(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness)
{
    vec3 H = normalize(V + L);

    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    float r = clamp(roughness, 0.04, 1.0);
    float alpha = r * r;

    // Dielectric F0 ~ 0.04, metallic uses albedo as F0
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3  F = fresnelSchlick(VdotH, F0);
    float D = D_GGX(NdotH, alpha);
    float G = G_SchlickSmith(NdotV, NdotL, alpha);

    vec3  spec = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-5);

    // Energy conservation: kD = (1 - F) * (1 - metallic)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 diff = (kD * albedo) * (1.0 / PI);

    // Return BRDF * cos term
    return (diff + spec) * NdotL;
}

bool isOccluded(vec3 origin, vec3 direction, float tmax)
{
    isShadowed = true;
    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
	traceRayEXT(topLevelAS, flags, 0xFF, 0, 0, 1, origin, EPSILON, direction, tmax, 1);
    return isShadowed;
}

vec3 getReflectedColor(vec3 origin, vec3 direction) 
{
    uint flags = gl_RayFlagsOpaqueEXT;
    traceRayEXT(topLevelAS, flags, 0xFF, 0, 0, 0, origin, EPSILON, direction, T_MAX, 0);
    return prd.hitValue;
}

// --- Legacy fallback (Lambert + simple Phong-like) ---
float legacy_specular(vec3 V, vec3 L, vec3 N, float roughness)
{
    vec3  H = normalize(V + L);
    float nDotH = max(dot(N, H), 0.0);
    float gloss = max(1.0 - roughness, 0.02);       // invert roughness
    float shin  = mix(8.0, 128.0, gloss);           // ad-hoc shininess
    return pow(nDotH, shin);
}

vec3 brdf_legacy(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness)
{
    float NdotL = max(dot(N, L), 0.0);
    // Mostly diffuse unless �metallic� is high (metal kills diffuse)
    vec3  diff  = albedo * (1.0 - metallic) * (NdotL / PI);
    float spec  = legacy_specular(V, L, N, roughness) * NdotL;
    // Mildly tint the spec with albedo for metals
    vec3  specC = mix(vec3(1.0), albedo, metallic) * spec * 0.25;
    return diff + specC;
}

void main()
{
    // Increment number of recursions
    bool computeReflection = true;
    prd.recursionDepth = prd.recursionDepth + 1;
    if (prd.recursionDepth == MAX_RECURSION - 1) {
        // we only want one more hit for the shadow
        computeReflection = false;
    }

    // Object data
    ObjDesc objResource = objDesc.i[gl_InstanceCustomIndexEXT];
    Indices indices = Indices(objResource.indexAddress);
    Vertices vertices = Vertices(objResource.vertexAddress);
    Material material = Material(objResource.materialAddress);
    MaterialRT mat = material.m;
  
    // Indices of the triangle
    ivec3 ind_init = indices.i[gl_PrimitiveID];
  
    // Vertex of the triangle
    Vertex v0 = vertices.v[ind_init.x];
    Vertex v1 = vertices.v[ind_init.y];
    Vertex v2 = vertices.v[ind_init.z];

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Get texture color
    vec2 uv0 = vec2(v0.uv_x, v0.uv_y);
    vec2 uv1 = vec2(v1.uv_x, v1.uv_y);
    vec2 uv2 = vec2(v2.uv_x, v2.uv_y);
    vec2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
	vec3 texColor = texture(ColImage2D[mat.textureID], uv).xyz * v0.color.xyz * mat.colorFactors.xyz;

    float metal;
    float roughness;
    if (USE_METAL_ROUGH_TEX) {
        // Get material data via metal-rough texture map
        vec4 matData = texture(MetalRoughImage2D[mat.textureID], uv);
        metal = matData.x;
        roughness = matData.y;
    } else {
        // Get material data via metal-rough factors
        metal = mat.metal_rough_factors.x;
        roughness = mat.metal_rough_factors.y;
    }

    // Computing the coordinates of the hit position
    const vec3 pos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

    // Computing the normal at hit position
    const vec3 nrm = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    const vec3 worldNrm = normalize(vec3(gl_ObjectToWorldEXT * vec4(nrm, 0.0)));  // Transforming the normal to world space

    int frameNumber = int(sceneData.data.x);
    vec3 outColor = vec3(0.0);

    

    // View vector points from surface to camera
    vec3 V = normalize(-gl_WorldRayDirectionEXT);

    // Lighting loop
    if (metal < 1.0 - EPSILON || !computeReflection) {
        for (int i = 0; i < pcRay.numLights; i++) {
            
            RenderLight l = Light(pcRay.lightAddress + i * RLstride).rl;
            vec3 lpos = l.position.xyz;
            float intensity = l.position.a;     // scalar intensity
            vec3 lcolor = l.color.xyz;          // light color (linear)
            int type = int(l.color.a);
            vec3 lv0 = l.v0.xyz;
            vec3 lv1 = l.v1.xyz;
            vec3 lv2 = lpos;

            if (type == POINT) {
                float dist = length(lpos - worldPos);
                vec3  L    = normalize(lpos - worldPos);
                // Both BRDFs scale by saturate(dot(N,L)), so a light below the
                // surface contributes nothing and its shadow ray is wasted.
                if (dot(worldNrm, L) <= 0.0) continue;
                bool  shadowed = isOccluded(worldPos, L, dist);
                if (!shadowed) {
                    float invDist2 = 1.0 / max(dist * dist, 1e-4);
                    vec3  radiance = lcolor * intensity * invDist2;

                    // Microfacet BRDF (diffuse+spec) * NdotL
                    vec3 contrib = (pcRay.useMicrofacet != 0)
                     ? brdf_ggx_smith(worldNrm, V, L, texColor, metal, roughness)
                     : brdf_legacy     (worldNrm, V, L, texColor, metal, roughness);
                    outColor += radiance * contrib;
                }

            } else if (type == AMBIENT) {
                // Ambient treated as diffuse only
                vec3 F0 = mix(vec3(0.04), texColor, metal);
                vec3 F  = fresnelSchlick(1.0, F0);
                vec3 kD = (vec3(1.0) - F) * (1.0 - metal);
                vec3 diffuse = kD * texColor * (1.0 / PI);
                outColor += diffuse * lcolor * intensity;

            } else if (type == DIRECTIONAL) {
                vec3  L = normalize(lpos); // direction stored in position.xyz
                if (dot(worldNrm, L) <= 0.0) continue;
                bool  shadowed = isOccluded(worldPos, L, T_MAX);
                if (!shadowed) {
                    vec3 radiance = lcolor * intensity;
                    vec3 contrib = (pcRay.useMicrofacet != 0)
                     ? brdf_ggx_smith(worldNrm, V, L, texColor, metal, roughness)
                     : brdf_legacy     (worldNrm, V, L, texColor, metal, roughness);
                    outColor += radiance * contrib;
                }

            } else if (type == AREA) {
                int samples = int(sceneData.data.y);
                for (int j = 0; j < samples; j++) {
                    vec2 rand = randomVec2(gl_LaunchIDEXT.xy, uint(frameNumber), uint(i) * 64u + uint(j));
                    if (rand.x + rand.y > 1.0) { rand = vec2(1.0) - rand; }
                    vec3 samplePoint = lv0 + (rand.x * (lv1 - lv0)) + (rand.y * (lv2 - lv0));

                    float dist = length(samplePoint - worldPos);
                    vec3  L    = normalize(samplePoint - worldPos);
                    if (dot(worldNrm, L) <= 0.0) continue;
                    bool  shadowed = isOccluded(worldPos, L, dist);
                    if (!shadowed) {
                        vec3 radiance = lcolor * intensity; // treat intensity as already scaled for area
                        vec3 contrib = (pcRay.useMicrofacet != 0)
                         ? brdf_ggx_smith(worldNrm, V, L, texColor, metal, roughness)
                         : brdf_legacy     (worldNrm, V, L, texColor, metal, roughness);
                        outColor += (radiance * contrib) / float(samples);
                    }
                }
            }
        }
    }

    // Calculate reflected color if reflective
    if (metal > EPSILON && computeReflection) {
        vec3 reflectedDir = reflect(gl_WorldRayDirectionEXT, worldNrm);
        outColor = (getReflectedColor(worldPos, reflectedDir) * metal) + (outColor * (1.0 - metal));
    }
    
    // === Volumetric over camera to first-hit segment ===
    // t from camera to this surface (GL_EXT_ray_tracing builtin)
    float tHit = gl_HitTEXT;  // distance in world units

    vec3 sig_t = sigma_t();
    vec3 Tr = exp(-sig_t * tHit);

    // Constant emission along the segment
    vec3 E = vec3(U_EMISSION);
    vec3 L_emis = E * safeDiv((vec3(1.0) - Tr), sig_t);

    // Apply absorption to surface light, add emission along view path
    outColor = Tr * outColor + L_emis;

    prd.hitValue = outColor;

    // Written after the reflection ray above has returned, so a reflected hit's
    // geometry never outlives the primary hit that spawned it.
    prd.worldNormal = worldNrm;
    prd.hitT = tHit;
    prd.instanceID = gl_InstanceCustomIndexEXT;
}

