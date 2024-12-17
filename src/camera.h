#include "SDL_events.h"
#include <vk_types.h>

class Camera {
public:
    glm::vec3 velocity;
    glm::vec3 position;
    // vertical rotation
    float pitch { 0.f };
    // horizontal rotation
    float yaw { 0.f };

    float fastSpeed = 3.0f;
    float slowSpeed = 1.0f;
    float normalSpeed = 0.025f;
    float currentSpeed = normalSpeed;

    glm::mat4 getViewMatrix() const;
    glm::mat4 getRotationMatrix() const;
    glm::vec3 getViewDirection() const;

    void processSDLEvent(SDL_Event& e);

    void update();
};
