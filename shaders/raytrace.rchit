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

// Information of a obj model when referenced in a shader
struct ObjDesc {
    uint64_t vertexAddress; // Address of the vertex buffer
    uint64_t indexAddress; // Address of the index buffer
    uint64_t materialAddress; // Address of the material buffer
};

layout(location = 0) rayPayloadInEXT hitPayload prd;
hitAttributeEXT vec2 attribs;

layout(buffer_reference, scalar) buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) buffer Indices { ivec3 i[]; };
layout(buffer_reference, scalar) buffer Material { MaterialRT m; };
layout(set = 2, binding = 0, scalar) buffer ObjDesc_ { ObjDesc i[]; } objDesc;

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };

void main()
{
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

    // Computing the coordinates of the hit position
    const vec3 pos = v0.position * barycentrics.x + v1.position * barycentrics.y + v2.position * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

    // Computing the normal at hit position
    const vec3 nrm = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    const vec3 worldNrm = normalize(vec3(gl_ObjectToWorldEXT * vec4(nrm, 0.0)));  // Transforming the normal to world space

    // Lighting loop
    prd.hitValue = vec3(0.0, 0.0, 0.0);
    int numLights = int(sceneData.numLights.x);

    // insert for loop
    for (int i = 0; i < numLights; i++) {

        // Unpack light info
        RenderLight l = sceneData.lights[i];
        vec3 lpos = l.position.xyz;
        float intensity = l.position.a;
        vec3 lcolor = l.color.xyz;
        int type = int(l.color.a);
        vec3 color = vec3(.5, .5, .5); // replace with texture color

        // Calculate by type
        if (type == POINT) {
            float dist = length(lpos - worldPos);
		    vec3 lightDir = normalize(lpos - worldPos);
		    float lightAmt = max(dot(nrm, lightDir), 0.0);
		    vec3 diffuse = lightAmt * intensity * lcolor / (dist * dist);
		    prd.hitValue += vec3(color * diffuse);
	    } else if (type == AMBIENT) {
		    prd.hitValue += vec3(color * intensity * lcolor);
	    } else if (type == DIRECTIONAL) {
		    vec3 lightDir = normalize(lpos);
		    float lightAmt = max(dot(nrm, lightDir), 0.0);
		    vec3 diffuse = lightAmt * intensity * lcolor;
		    prd.hitValue += vec3(color * diffuse);
	    }
    }
    prd.hitValue = mat.metal_rough_factors.xyz;
}

