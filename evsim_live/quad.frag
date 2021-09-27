#version 150 core

in vec2 Texcoord;

uniform sampler2D texFramebuffer;
uniform int doGammaCorr;
uniform int invertY;

uniform vec2 offset;
uniform vec2 scale;

out vec4 outColor;

void main(void) {
//    outColor = vec4(1.0, 1.0, 1.0, 1.0);
//    outColor = vec4(Texcoord.x, Texcoord.y, 1.0, 1.0);
    vec2 texcoord = Texcoord;
    if (invertY > 0) {
        texcoord.y = 1-texcoord.y;
    }
    outColor = texture(texFramebuffer, (texcoord-vec2(0.5))/scale+vec2(0.5)+offset);
    if (doGammaCorr == 1) {
        outColor.rgb = pow(outColor.rgb, vec3(1.0/2.2)); // forward correction
    } else if (doGammaCorr == 2) {
        outColor.rgb = pow(outColor.rgb, vec3(2.2/1.0)); // inverse correction
    }
}
