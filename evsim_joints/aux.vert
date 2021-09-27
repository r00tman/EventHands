#version 150 core
in vec3 position;
in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec4 projNormal;

void main(void) {
//    gl_Position = vec4(position.xy, 0.0, 1.0);
    gl_Position = proj * view * model * vec4(position, 1.0);
//    gl_Position = model * vec4(position, 1.0);
    projNormal = normalize(model * vec4(normal, 0.0));
}
