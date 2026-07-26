// Octahedral normal encoding: https://jcgt.org/published/0003/02/01/
// Kept free of descriptor declarations so compute passes can include it without
// pulling in the ray tracing descriptor sets.

vec2 oct_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return n.xy;
}

vec3 oct_decode(vec2 e) {
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}
