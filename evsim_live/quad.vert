#version 150 core

in vec2 position;

out vec2 Texcoord;

void main(void) {
    gl_Position = vec4(position, 0.0, 1.0);
    Texcoord = (position+1.0)/2.0;
}
