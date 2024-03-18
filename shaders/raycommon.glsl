struct RenderLight {
    vec4 position; // a is intensity
    vec4 color; // a is type, 0 is point, 1 is ambient
};

layout(set = 1, binding = 0) uniform SceneData {   
	mat4 view;
	mat4 proj;
	mat4 viewproj;
    vec4 numLights; // x is num lights
    RenderLight lights[16];
} sceneData;

struct hitPayload
{
  vec3 hitValue;
};