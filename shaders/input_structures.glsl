
struct RenderLight {
    vec4 position; // a is intensity
    vec4 color; // a is type, 0 is point, 1 is ambient
};

layout(set = 0, binding = 0) uniform SceneData {   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
    vec4 numLights; // x is num lights
    RenderLight lights[4];
} sceneData;

layout(set = 1, binding = 0) uniform GLTFMaterialData {   
	vec4 colorFactors;
	vec4 metal_rough_factors;
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;