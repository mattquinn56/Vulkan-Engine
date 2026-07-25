#include "camera.h"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/transform.hpp"

glm::mat4 Camera::get_view_matrix() const {
    // Invert the camera transform to move the world relative to the camera.
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = get_rotation_matrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::get_rotation_matrix() const {
    // Apply yaw before pitch for an FPS-style camera.
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{1.f, 0.f, 0.f});
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{0.f, -1.f, 0.f});

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

glm::vec3 Camera::get_view_direction() const {
    glm::vec3 direction;
    direction.x = sin(yaw);
    direction.y = sin(pitch);
    direction.z = -cos(yaw);
    return direction;
}

void Camera::process_sdl_event(SDL_Event& e) {
    if (e.type == SDL_MOUSEMOTION) {
        yaw += static_cast<float>(e.motion.xrel) / 200.0f;
        pitch -= static_cast<float>(e.motion.yrel) / 200.0f;
    }
}

void Camera::update() {
    const Uint8* keystate = SDL_GetKeyboardState(nullptr);

    if (keystate[SDL_SCANCODE_LSHIFT] || keystate[SDL_SCANCODE_RSHIFT]) {
        currentSpeed = fastSpeed;
    } else if (keystate[SDL_SCANCODE_LCTRL] || keystate[SDL_SCANCODE_RCTRL]) {
        currentSpeed = slowSpeed;
    } else {
        currentSpeed = normalSpeed;
    }

    velocity = glm::vec3(0.f);
    if (keystate[SDL_SCANCODE_W])
        velocity.z = -currentSpeed;
    if (keystate[SDL_SCANCODE_S])
        velocity.z = currentSpeed;
    if (keystate[SDL_SCANCODE_A])
        velocity.x = -currentSpeed;
    if (keystate[SDL_SCANCODE_D])
        velocity.x = currentSpeed;
    if (keystate[SDL_SCANCODE_E])
        velocity.y = currentSpeed;
    if (keystate[SDL_SCANCODE_Q])
        velocity.y = -currentSpeed;

    // Horizontal movement is camera-relative; vertical is world-space up
    glm::mat4 cameraRotation = get_rotation_matrix();
    glm::vec3 horizVelocity(velocity.x, 0.f, velocity.z);
    position += glm::vec3(cameraRotation * glm::vec4(horizVelocity, 0.f)) * 0.5f;
    position.y += velocity.y * 0.5f;
}
