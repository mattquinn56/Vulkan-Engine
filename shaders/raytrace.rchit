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
const float EPSILON = .001;
const float T_MAX = 10000.0;
const int MAX_RECURSION = 4; // should be the same as MAX_RECURSION in vk_raytracer.h

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
layout(set = 3, binding = 0) uniform sampler2D[] ColImage2D;
layout(set = 3, binding = 1) uniform sampler2D[] MetalRoughImage2D;

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

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

float computeSpecularIntensity(vec3 viewDir, vec3 lightDir, vec3 normal, float roughness)
{
    /*
    vec3 reflectDir = reflect(-lightDir, normal); // Reflect function calculates the reflection direction
    viewDir = normalize(viewDir); // Normalize the view direction vector

    float specAngle = max(dot(reflectDir, viewDir), 0.0);
    float shininess = 1.0 / roughness; // Shininess is the inverse of roughness
    float specular = pow(specAngle, shininess);

    return specular;
    */
    vec3 halfVec = normalize(lightDir + viewDir);
    float nDotVP = max(0.0, dot(normal, normalize(lightDir) ));	
	float nDotHV = max(0.0, dot(normal, normalize(halfVec) ));
	
    float pf;
	if(nDotVP == 0.0) {
		pf = 0.0;
    } else {
        pf = pow(nDotHV, 1.0 / roughness);
	}

    return pf;
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

    // Get material data via metal-rough texture map
    //vec4 matData = texture(MetalRoughImage2D[mat.textureID], uv);
    //float metal = matData.x;
    //float roughness = matData.y;
    
    // Get material data via metal-rough factors
    float metal = mat.metal_rough_factors.x;
    float roughness = 0.0;//mat.metal_rough_factors.y;

    // Computing the coordinates of the hit position
    const vec3 pos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

    // Computing the normal at hit position
    const vec3 nrm = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    const vec3 worldNrm = normalize(vec3(gl_ObjectToWorldEXT * vec4(nrm, 0.0)));  // Transforming the normal to world space

    vec3 outColor = vec3(0.0);

    // Lighting loop
    if (metal < 1.0 - EPSILON || !computeReflection) {
        for (int i = 0; i < pcRay.numLights; i++) {
            
            // Unpack light info
            RenderLight l = Light(pcRay.lightAddress + i * RLstride).rl;
            vec3 lpos = l.position.xyz;
            float intensity = l.position.a;
            vec3 lcolor = l.color.xyz;
            int type = int(l.color.a);
            vec3 v0 = l.v0.xyz;
            vec3 v1 = l.v1.xyz;
            vec3 v2 = lpos;

            // Calculate by type
            if (type == POINT) {
                float dist = length(lpos - worldPos);
		        vec3 lightDir = normalize(lpos - worldPos);
                bool shadowed = isOccluded(worldPos, lightDir, dist);
                if (!shadowed) {
                    // compute diffuse
		            float lightAmt = max(dot(worldNrm, lightDir), 0.0);
		            vec3 diffuse = lightAmt * intensity * lcolor / (dist * dist);
				    outColor += vec3(texColor * diffuse);
                    
                    // compute specular
					float specular = computeSpecularIntensity(gl_WorldRayDirectionEXT, lightDir, worldNrm, roughness);
					outColor += specular * lcolor * intensity;
			    }

	        } else if (type == AMBIENT) {
		        outColor += vec3(texColor * intensity * lcolor);

	        } else if (type == DIRECTIONAL) {
		        vec3 lightDir = normalize(lpos);
                bool shadowed = isOccluded(worldPos, lightDir, T_MAX);
                if (!shadowed) {
                    // compute diffuse
		            float lightAmt = max(dot(worldNrm, lightDir), 0.0);
		            vec3 diffuse = lightAmt * intensity * lcolor;
				    outColor += vec3(texColor * diffuse);

                    // compute specular
					float specular = computeSpecularIntensity(gl_WorldRayDirectionEXT, lightDir, worldNrm, roughness);
					outColor += specular * lcolor * intensity;
                }
	        } else if (type == AREA) {
                outColor += vec3(0.0);
            }
        }
    }

    // Calculate reflected color if reflective
    if (metal > EPSILON && computeReflection) {
        vec3 reflectedDir = reflect(gl_WorldRayDirectionEXT, worldNrm);
        outColor = (getReflectedColor(worldPos, reflectedDir) * metal) + (outColor * (1.0 - metal));
    }
    
    prd.hitValue = outColor;
}

