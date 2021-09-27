#version 150 core

in vec4 projNormal;

uniform vec4 lightdir;
uniform vec3 lightcol;
uniform vec4 lightdir2;
uniform vec3 lightcol2;

out vec4 outColor;

void main(void) {
    float light = max(0, dot(projNormal, normalize(lightdir)));
//    vec4 -lightdir2 = vec4(-1.2, -1.2, -1.2, 0.0);
    float light2 = max(0, dot(projNormal, normalize(lightdir2)));
    vec3 pre = vec3(0.0, 0.0, 0.0);
    pre += light * lightcol;//vec3(1.0, 0.7, 0.5);
    pre += light2 * lightcol2;//vec3(0.5, 0.7, 1.0)*0.3;
    pre += vec3(0.05, 0.05, 0.1);
//    pre = pre * 1.4;
//    albedo = pow(albedo, vec4(1.4/1.0));
//    pre = log(pre+0.01)+1;
//    pre = pow(pre, vec3(1.0/2.2));
    outColor = vec4(pre, 1.0);
}
