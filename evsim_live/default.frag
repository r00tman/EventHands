#version 150 core
//uniform sampler2D qt_Texture0;
//varying vec4 qt_TexCoord0;

in vec4 projNormal;
in vec2 Texcoord;

uniform sampler2D texHand;
uniform vec4 lightdir;
uniform vec3 lightcol;
uniform vec4 lightdir2;
uniform vec3 lightcol2;

out vec4 outColor;

void main(void) {
//    gl_FragColor = texture2D(qt_Texture0, qt_TexCoord0.st);
//    outColor = vec4(1.0, 1.0, 1.0, 1.0);
//    outColor = vec4(triangleColor, 1.0);
//    vec4 -lightdir = vec4(1.2, -1.2, 1.2, 0.0);
//    vec4 lightdir = -vec4(1, 0, 0, 0.0);
    float light = max(0, dot(projNormal, normalize(lightdir)));
//    vec4 -lightdir2 = vec4(-1.2, -1.2, -1.2, 0.0);
    float light2 = max(0, dot(projNormal, normalize(lightdir2)));
    vec3 pre = vec3(0.0, 0.0, 0.0);
    pre += light * lightcol;//vec3(1.0, 0.7, 0.5);
    pre += light2 * lightcol2;//vec3(0.5, 0.7, 1.0)*0.3;
    pre += vec3(0.05, 0.05, 0.1);
    pre = pre * 1.4;
    vec4 albedo = texture(texHand, Texcoord);
//    albedo = pow(albedo, vec4(1.4/1.0));
//    pre = log(pre+0.01)+1;
//    pre = pow(pre, vec3(1.0/2.2));
    outColor = vec4(pre*albedo.rgb, 1.0);
//    outColor = vec4(Texcoord, 0.0, 1.0);
//    outColor = vec4(triangleColor*light, 1.0);
}
