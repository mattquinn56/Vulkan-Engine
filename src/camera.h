#include "SDL_events.h"
#include <vk_types.h>

class Camera
{
  public:
    glm::vec3 velocity;
    glm::vec3 position;
    // vertical rotation
    float pitch{0.f};
    // horizontal rotation
    float yaw{0.f};

    float fastSpeed = 0.25f;
    float normalSpeed = 0.025f;
    float slowSpeed = 0.0025f;
    float currentSpeed = normalSpeed;

    glm::mat4 get_view_matrix() const;
    glm::mat4 get_rotation_matrix() const;
    glm::vec3 get_view_direction() const;

    void process_sdl_event(SDL_Event& e);

    void update();
};
