import pymunk


def limit_velocity(body, gravity, damping, dt, max_velocity=100):
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    l = body.velocity.length
    if l > max_velocity:
        scale = max_velocity / l
        body.velocity = body.velocity * scale
