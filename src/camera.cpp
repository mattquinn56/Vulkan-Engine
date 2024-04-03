#include "camera.h"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/transform.hpp"

glm::mat4 Camera::getViewMatrix() const
{
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    //  so we will create the camera model matrix and invert
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix() const
{
    // fairly typical FPS style camera. we join the pitch and yaw rotations into
    // the final rotation matrix

    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3 { 1.f, 0.f, 0.f });
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3 { 0.f, -1.f, 0.f });

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

glm::vec3 Camera::getViewDirection() const
{
    /*
    glm::vec3 viewDir = glm::vec3();
	return glm::vec3(getViewMatrix() * glm::vec4(0, 0, -1, 0));
    viewDir.x = -viewDir.x;
    viewDir.y = -viewDir.y;
    return viewDir;
    */

    // Calculate the direction vector
    glm::vec3 direction;
    direction.x = sin(yaw);
    direction.y = sin(pitch);
    direction.z = -cos(yaw);
    return direction;
}

void Camera::processSDLEvent(SDL_Event& e)
{
    if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_LSHIFT || e.key.keysym.sym == SDLK_RSHIFT) {
            currentSpeed = fastSpeed;
        }
        else if (e.key.keysym.sym == SDLK_LCTRL || e.key.keysym.sym == SDLK_RCTRL) {
            currentSpeed = slowSpeed;
        }

        if (e.key.keysym.sym == SDLK_w) { velocity.z = -currentSpeed; }
        if (e.key.keysym.sym == SDLK_s) { velocity.z = currentSpeed; }
        if (e.key.keysym.sym == SDLK_a) { velocity.x = -currentSpeed; }
        if (e.key.keysym.sym == SDLK_d) { velocity.x = currentSpeed; }
    }

    if (e.type == SDL_KEYUP) {
        if (e.key.keysym.sym == SDLK_LSHIFT || e.key.keysym.sym == SDLK_RSHIFT) {
            currentSpeed = normalSpeed;
        }
        else if (e.key.keysym.sym == SDLK_LCTRL || e.key.keysym.sym == SDLK_RCTRL) {
            currentSpeed = normalSpeed;
        }

        if (e.key.keysym.sym == SDLK_w || e.key.keysym.sym == SDLK_s) { velocity.z = 0; }
        if (e.key.keysym.sym == SDLK_a || e.key.keysym.sym == SDLK_d) { velocity.x = 0; }
    }

    if (e.type == SDL_MOUSEMOTION) {
        yaw += static_cast<float>(e.motion.xrel) / 200.0f;
        pitch -= static_cast<float>(e.motion.yrel) / 200.0f;
    }
}

void Camera::update()
{
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.5f, 0.f));
}
