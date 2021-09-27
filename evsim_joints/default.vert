#version 150 core
const int N_COMP = 52;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

in vec4 weights[N_COMP/4];

out vec4 projNormal;
out vec2 Texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

// gets 4 input components, outputs 3
uniform mat4x3 A[N_COMP];


void main(void) {
    mat4x3 T = mat4x3(0);
    for (int j = 0; j < 52/4; ++j) {
        T += A[4*j+0] * weights[j].x;
        T += A[4*j+1] * weights[j].y;
        T += A[4*j+2] * weights[j].z;
        T += A[4*j+3] * weights[j].w;
    }

    vec3 output = T * vec4(position, 1.0);
    vec3 outputNormal = T * vec4(normal, 0.0);

    // crop just the arm
    if (position.x > -0.1) {
        output *= 0;
        output.x = -0.1;
    }
    if (position.y < 0) {
        output *= 0;
        output.x = -0.1;
    }
    if (position.y > 0.3) {
        output *= 0;
        output.x = -0.1;
    }

    gl_Position = proj * view * model * vec4(output, 1.0);
    projNormal = normalize(model * vec4(outputNormal, 0.0));
    Texcoord = vec2(texcoord.x, 1-texcoord.y);
}
