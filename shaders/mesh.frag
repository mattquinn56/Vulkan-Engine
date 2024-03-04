#version 450

#extension GL_GOOGLE_include_directive : require
#include "input_structures.glsl"

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inPos;

layout (location = 0) out vec4 outFragColor;

const vec3 sunlightDir = vec3(0.3f,1.f,0.3f);

// temporarily hardcoding the light values
const vec3 pointLightPos = vec3(10.0f, 10.0f, 0.0f);
const float pointLightIntensity = 10000.0f;
const vec3 pointLightColor = vec3(0.7f, .85f, 1.0f);
const float ambientLightIntensity = 0.f;
const vec3 ambientLightColor = vec3(1.0f, 1.0f, 1.0f);

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

void main() 
{
    // old lighting algorithm
	//float lightValue = max(dot(inNormal, sunlightDir), 0.1f);
	//vec3 irradiance = calcIrradiance(inNormal); 
	//vec3 color = inColor * texture(colorTex, inUV).xyz;
	//outFragColor = vec4(color * lightValue + color * irradiance.x * vec3(0.2f) ,1.0f);

    // new lighting algorithm
    vec3 color = inColor * texture(colorTex, inUV).xyz;

    vec3 normal = normalize(inNormal);
    float dist = length(pointLightPos - inPos);
    vec3 lightDir = normalize(pointLightPos - inPos);
    float lightAmt = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = lightAmt * pointLightIntensity * pointLightColor / (dist * dist);

    vec3 ambient = ambientLightIntensity * ambientLightColor;

    outFragColor = vec4(color * (diffuse + ambient), 1.0);
}

