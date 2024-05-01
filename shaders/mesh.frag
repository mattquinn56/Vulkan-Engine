#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inPos;

struct Vertex {

	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer { 
	Vertex vertices[];
};

layout(buffer_reference, std430) buffer Light { RenderLight rl; };

//push constants block
layout( push_constant ) uniform constants {
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
	uint64_t lightAddress;
    uint numLights;
} PushConstants;

layout (location = 0) out vec4 outFragColor;

const int POINT = 0;
const int AMBIENT = 1;
const int DIRECTIONAL = 2;

struct SHCoefficients {
    vec3 l00, l1m1, l10, l11, l2m2, l2m1, l20, l21, l22;
};

const SHCoefficients grace = SHCoefficients(
    vec3( 0.3623915,  0.2624130,  0.2326261 ),
    vec3( 0.1759131,  0.1436266,  0.1260569 ),
    vec3(-0.0247311, -0.0101254, -0.0010745 ),
    vec3( 0.0346500,  0.0223184,  0.0101350 ),
    vec3( 0.0198140,  0.0144073,  0.0043987 ),
    vec3(-0.0469596, -0.0254485, -0.0117786 ),
    vec3(-0.0898667, -0.0760911, -0.0740964 ),
    vec3( 0.0050194,  0.0038841,  0.0001374 ),
    vec3(-0.0818750, -0.0321501,  0.0033399 )
);

vec3 calcIrradiance(vec3 nor) {
    const SHCoefficients c = grace;
    const float c1 = 0.429043;
    const float c2 = 0.511664;
    const float c3 = 0.743125;
    const float c4 = 0.886227;
    const float c5 = 0.247708;
    return (
        c1 * c.l22 * (nor.x * nor.x - nor.y * nor.y) +
        c3 * c.l20 * nor.z * nor.z +
        c4 * c.l00 -
        c5 * c.l20 +
        2.0 * c1 * c.l2m2 * nor.x * nor.y +
        2.0 * c1 * c.l21  * nor.x * nor.z +
        2.0 * c1 * c.l2m1 * nor.y * nor.z +
        2.0 * c2 * c.l11  * nor.x +
        2.0 * c2 * c.l1m1 * nor.y +
        2.0 * c2 * c.l10  * nor.z
    );
}

void main() {

    // up to 16 lights
    outFragColor = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 color = inColor * texture(colorTex, inUV).xyz;
    vec3 normal = normalize(inNormal);
    for (int i = 0; i < PushConstants.numLights; i++) {
        RenderLight l = Light(PushConstants.lightAddress + i * RLstride).rl;
        float power = l.position.a;
        int type = int(l.color.a);

	    if (type == POINT) {
            float dist = length(l.position.xyz - inPos);
		    vec3 lightDir = normalize(l.position.xyz - inPos);
		    float lightAmt = max(dot(normal, lightDir), 0.0);
		    vec3 diffuse = lightAmt * power * l.color.xyz / (dist * dist);
		    outFragColor.xyz += vec3(color * diffuse);
	    } else if (type == AMBIENT) {
		    outFragColor.xyz += vec3(color * power * l.color.xyz);
	    } else if (type == DIRECTIONAL) {
		    vec3 lightDir = normalize(l.position.xyz);
		    float lightAmt = max(dot(normal, lightDir), 0.0);
		    vec3 diffuse = lightAmt * power * l.color.xyz;
		    outFragColor.xyz += vec3(color * diffuse);
	    }
    }
}

