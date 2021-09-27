#include <cstdint>
#include <cmath>
#include <ctime>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <experimental/filesystem>

#define GLEW_STATIC
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "clipp.h"

#include "log.h"
#include "config.h"

#include "evc.h"

#include "shader.h"
#include "model.h"
#include "manomodel.h"
#include "framebuffer.h"
#include "mano.h"
#include "metadatawriter.h"
#include "texture.h"
#include "manotexture.h"


void run(SDL_Window *window, const std::string &fn) {

    xt::random::seed(time(0));

    // load background textures
    std::vector<std::unique_ptr<Texture> > backgroundTextures;
    for (const auto &entry : std::experimental::filesystem::directory_iterator("backgrounds")) {
        backgroundTextures.emplace_back(new Texture(entry.path()));
    }
    int backgroundTextureIdx = 0;

    ManoTexture handTexture;

    glEnable(GL_DEPTH_TEST);

    // load the quad shader used for background and post-processing and create the corresponding quad
    Shader quadShader("quad.vert", "quad.frag");
    quadShader.use();

    std::unique_ptr<Model> quad = new_quad([&]() {
        GLint posAtrib = quadShader.attrib("position");
        glVertexAttribPointer(posAtrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(posAtrib);
    });

    // load the shader for the debug arrow and the corresponding model
    Shader auxShader("aux.vert", "aux.frag");
    std::unique_ptr<Model> arrow;
    {
        std::vector<float> vertices;
        std::vector<GLuint> elements;
        load_obj("arrow.obj", vertices, elements);
        arrow = std::unique_ptr<Model>(new Model(vertices, elements, false, [&]() {
            GLint posAtrib = auxShader.attrib("position");
            glVertexAttribPointer(posAtrib, 3, GL_FLOAT, GL_FALSE,
                                  6*sizeof(float), 0);
            glEnableVertexAttribArray(posAtrib);

            GLint normAtrib = auxShader.attrib("normal");
            glVertexAttribPointer(normAtrib, 3, GL_FLOAT, GL_FALSE,
                                  6*sizeof(float), (void*)(3*sizeof(float)));
            glEnableVertexAttribArray(normAtrib);
        }));
    }


    // load the arm shader and the corresponding model
    Shader modelShader("default.vert", "default.frag");
    modelShader.use();
    Mano m;

    std::unique_ptr<ManoModel> model;

    {
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<GLuint> elements;
        std::vector<float> weights;
        std::vector<float> texcoords;
        std::vector<glm::mat4x3> A;

        // generate the arm model
        m.generate(vertices, normals, elements, weights, texcoords, A);

        // create the OpenGL model from the generated data
        model = std::unique_ptr<ManoModel>(new ManoModel(vertices, normals, elements, weights, texcoords, A,
                                                         [&] (GLuint vbo, GLuint nbo, GLuint wbo, GLuint tbo) {
            // attach all the buffers to the shader
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            GLint posAtrib = modelShader.attrib("position");
            glVertexAttribPointer(posAtrib, 3, GL_FLOAT, GL_FALSE,
                                  3*sizeof(float), 0);
            glEnableVertexAttribArray(posAtrib);

            glBindBuffer(GL_ARRAY_BUFFER, nbo);
            GLint normAtrib = modelShader.attrib("normal");
            glVertexAttribPointer(normAtrib, 3, GL_FLOAT, GL_FALSE,
                                  3*sizeof(float), 0);
            glEnableVertexAttribArray(normAtrib);

            glBindBuffer(GL_ARRAY_BUFFER, wbo);
            GLint weightsAttrib = modelShader.attrib("weights");
            for (int i = 0; i < 52/4; ++i) {
                glVertexAttribPointer(weightsAttrib+i, 4, GL_FLOAT, GL_FALSE,
                                      52*sizeof(float), (void*)((4*i)*sizeof(float)));
                glEnableVertexAttribArray(weightsAttrib+i);
            }

            glBindBuffer(GL_ARRAY_BUFFER, tbo);
            GLint texcoordAtrib = modelShader.attrib("texcoord");
            glVertexAttribPointer(texcoordAtrib, 2, GL_FLOAT, GL_FALSE,
                                  2*sizeof(float), 0);
            glEnableVertexAttribArray(texcoordAtrib);
        }));
    }


    Framebuffer fb(WINDOW_WIDTH, WINDOW_HEIGHT);

    evcData_t evcData;

    if (PLAYBACK.length() > 0) {
        // don't write the simulated output to file on playback
        init_evc(&evcData, fb.texColorBuffer, WINDOW_WIDTH, WINDOW_HEIGHT, NULL);
    } else {
        std::string evcFilename = fn+".evc";
        init_evc(&evcData, fb.texColorBuffer, WINDOW_WIDTH, WINDOW_HEIGHT, evcFilename.c_str());
    }

    uint8_t *videoRenderFrame;
    if (VIDEO_RENDER) {
        videoRenderFrame = new uint8_t[WINDOW_WIDTH*WINDOW_HEIGHT*3];
    }

    FILE *jointFile = nullptr;
    if (JOINT_OUTPUT.length() > 0) {
        jointFile = fopen(JOINT_OUTPUT.c_str(), "w");
    }

    // frame counter
    int cntr = 0;

    std::unique_ptr<MetadataWriter> writer;
    if (PLAYBACK.length() > 0) {
        // don't write the simulated output to file on playback
    } else {
        // n_comps*mano param, 3*pos, 3*rot
        writer.reset(new MetadataWriter(fn+".meta", m.N_COMPS/2+6));
    }

    glm::mat4 neutral_trans = glm::mat4(1.0);
    if (PLAYBACK.length() > 0) {
        // on playback we need to replicate exact hand position
        // and rotation. so, we do that by neutralizing the whole
        // base hand to identity with inverse neutral transform,
        // and then transforming to the position it should be at
        neutral_trans = glm::mat4(1.0f);
        neutral_trans = glm::translate(neutral_trans, m.get_hand_position());
        glm::vec3 rot = m.get_hand_rotation();
        neutral_trans = glm::rotate(neutral_trans, glm::length(rot), glm::normalize(rot));
//        neutral_trans = glm::inverse(neutral_trans);
    }

    // set camera matrices
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), WINDOW_WIDTH * 1.0f / WINDOW_HEIGHT, 0.1f, 10.0f);
    glUniformMatrix4fv(modelShader["proj"], 1, GL_FALSE, glm::value_ptr(proj));
//    glm::vec3 eyePos = glm::rotate(glm::mat4(1.0f), glm::radians(130.0f), glm::vec3(0.0, 1.0f, 0.0f))*
//                       glm::vec4(1.2f, 1.2f, 1.2f, 1.0f);
//    glm::vec3 eyePos = glm::vec3(-0.3f, 0.16f, 0.4f);
    glm::vec3 eyePos = glm::vec3(-0.45f, -0.05f, 0.4f);
//    glm::vec3 eyePos = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::mat4 view = glm::lookAt(
        eyePos,
        glm::vec3(eyePos.x-0.2f, eyePos.y+0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(modelShader["view"], 1, GL_FALSE, glm::value_ptr(view));

    // light parameters
    glm::vec4 lightdir{-1.2, 1.2, 1.2, 0.0}, lightdir2{1.2, 1.2, -1.2, 0.0};
    glm::vec3 lightcol{1.0, 0.7,  0.5}, lightcol2{0.5, 0.7, 1.0};
    lightcol2 *= 0.3;

    // background position/scale
    glm::vec2 bgOffset{0.0, 0.0};
    glm::vec2 bgScale{1.0, 1.0};

    // runtime flags
    bool showCuda = false;
    bool showDebugArrow = false;
    bool doGammaCorr = true;
    bool screenRender = true;
    bool animate = true;
    bool changeBetas = false;
    bool changeTexture = false;
    bool computeManoOnCuda = true;
    bool changeLights = false;
    bool changeBg = false;
    bool changeC = false;
    float C = 0.5f;

    if (OFFSCREEN_RENDER) {
        screenRender = false;
        m.dt = 1.L/1000.L;
    }

    if (FIFO) {
        m.dt = 1.L/60.L;
    }

    if (VIDEO_RENDER) {
        m.dt = 1.L/1000.L;
    }

    if (DISPLAY_FPS > 0) {
        m.dt = 1.L/DISPLAY_FPS;
    }

//    m.dt = 1/1000.;
//    screenRender = false;

    uint32_t start = SDL_GetTicks();

    glClearColor(0.025, 0.025, 0.05, 1.0);
    while (true) {
        SDL_Event e;
        if (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                break;
            }
            if (e.type == SDL_KEYUP) {
                if (e.key.keysym.sym == SDLK_v) {
                    showCuda = !showCuda;
                } else if (e.key.keysym.sym == SDLK_g) {
                    doGammaCorr = !doGammaCorr;
                } else if (e.key.keysym.sym == SDLK_r) {
                    screenRender = !screenRender;
                    start = SDL_GetTicks();
                    cntr = 0;
                } else if (e.key.keysym.sym == SDLK_a) {
                    animate = !animate;
                } else if (e.key.keysym.sym == SDLK_b) {
                    changeBetas = true;
                } else if (e.key.keysym.sym == SDLK_c) {
                    computeManoOnCuda = !computeManoOnCuda;
                } else if (e.key.keysym.sym == SDLK_d) {
                    showDebugArrow = !showDebugArrow;
                } else if (e.key.keysym.sym == SDLK_t) {
                    changeTexture = true;
                } else if (e.key.keysym.sym == SDLK_l) {
                    changeLights = true;
                } else if (e.key.keysym.sym == SDLK_o) {
                    changeBg = true;
                } else if (e.key.keysym.sym == SDLK_i) {
                    changeBetas = true;
                    changeLights = true;
                    changeTexture = true;
                    changeBg = true;
                    changeC = true;
                } else if (e.key.keysym.sym == SDLK_0) {
                    C *= 1.5;
                } else if (e.key.keysym.sym == SDLK_9) {
                    C /= 1.5;
                } else if (e.key.keysym.sym == SDLK_8) {
                    m.dt *= 1.5;
                } else if (e.key.keysym.sym == SDLK_7) {
                    m.dt /= 1.5;
                } else if (e.key.keysym.sym == SDLK_6) {
                    m.dt = 1/1000.;
                } else if (e.key.keysym.sym == SDLK_5) {
                    m.dt = 1/60.;
                } else if (e.key.keysym.sym == SDLK_ESCAPE ||
                           e.key.keysym.sym == SDLK_q) {
                    break;
                }
            }
        }

        // once in 50 simulated seconds, rerandomize everything
        const long double PERIOD = 50;
        if (fmodl(m.t, PERIOD) < fmodl(m.t-m.dt+PERIOD, PERIOD) && PLAYBACK.length() == 0) {
            changeBetas = true;
            changeLights = true;
            changeTexture = true;
            changeBg = true;
            changeC = true;
        }

        // if asked to change the event generation threshold C, sample it from the normal distribution
        if (changeC) {
            xt::xarray<double> tmp = xt::random::randn({1ul}, 0.5, 0.02);
            C = tmp(0);
            changeC = false;
        }

        uint32_t now = SDL_GetTicks() - start;
        // draw the model onto the backbuffer
        fb.use();
        glDisable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);

        // background quad
        quad->use();
        quadShader.use();
        backgroundTextures[backgroundTextureIdx]->use(quadShader["texFramebuffer"]);
        glUniform1i(quadShader["doGammaCorr"], 2); // inverse correction
        glUniform1i(quadShader["invertY"], 1); // invert y

        if (changeBg) {
            backgroundTextureIdx = (rand()*1.0/RAND_MAX)*backgroundTextures.size();
            {
                xt::xarray<double> tmp = xt::random::rand<double>({2ul}, -0.1, 0.1);
                bgOffset.x = tmp[0];
                bgOffset.y = tmp[1];
            }
            {
                xt::xarray<double> tmp = xt::random::rand<double>({2ul}, 0.0, 0.1);
                tmp *= 0;
                bgScale.x = 1.3+tmp[0];
                bgScale.y = 1.3+tmp[1];
            }
            changeBg = false;
        }

        glUniform2f(quadShader["offset"], bgOffset.x, bgOffset.y);
        glUniform2f(quadShader["scale"], bgScale.x, bgScale.y);

        quad->draw();


        glEnable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);

        model->use();
        bool changed = false;
        if (changeBetas) {
            m.change_betas();
            changed = true;
            changeBetas = false;
        }
        if (animate) {
            m.change_pose(false);
            changed = true;
        }
        modelShader.use();
        if (changed) {
            if (computeManoOnCuda) {
                m.generate(model->vbo, model->nbo, model->ebo, model->A);
            } else {
                m.generate(model->vertices, model->normals, model->elements, model->weights, model->texcoords, model->A);
                model->update_vertices();
                model->update_normals();
            }
            auto sspos = glm::vec4(m.get_hand_position(), 1.0f);
            sspos = proj * view * sspos;
//            std::cerr << sspos.x << ' ' << sspos.y << ' ' << sspos.z << std::endl;
            if (sspos.x < -0.3 || sspos.x > 0.4 || sspos.y < -0.25 || sspos.y > 0.3 || sspos.z < 0.1) {
                m.change_pose(true);
            }
            model->update_shader(modelShader);
            if (PLAYBACK.length() > 0) {
                // on playback we need to replicate exact hand position
                // and rotation. so, we do that by neutralizing the whole
                // base hand to identity with inverse neutral transform,
                // and then transforming to the position it should be at
                neutral_trans = glm::mat4(1.0f);
                neutral_trans = glm::translate(neutral_trans, m.get_hand_position());
                glm::vec3 rot = m.get_hand_rotation();
                neutral_trans = glm::rotate(neutral_trans, glm::length(rot), glm::normalize(rot));
        //        neutral_trans = glm::inverse(neutral_trans);
            }
        }

        if (changeTexture) {
            handTexture.change();
            changeTexture = false;
        }

        // model world transform
        glm::mat4 trans = glm::mat4(1.0f);
        if (PLAYBACK.length() > 0) {
            glm::vec3 rot;
            if (FIFO) {
                assert(dynamic_cast<ManoFifoAnimator*>(m.animator));
                trans = glm::translate(trans, ((ManoFifoAnimator*)m.animator)->mytrans);
                rot = ((ManoFifoAnimator*)m.animator)->myrot;
            } else {
                assert(dynamic_cast<ManoFileAnimator*>(m.animator));
                trans = glm::translate(trans, ((ManoFileAnimator*)m.animator)->mytrans);
                rot = ((ManoFileAnimator*)m.animator)->myrot;
            }
            trans = glm::rotate(trans, glm::length(rot), glm::normalize(rot));

            trans = trans * glm::inverse(neutral_trans);
        } else {
//            trans = glm::rotate(trans, m.t * glm::radians(180.0f) / 8, glm::vec3(0.0f, 1.0f, 0.0f));
//            trans = glm::rotate(trans, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
//            trans = glm::translate(trans, glm::vec3(+4.3, -0.9, 0)+m.get_trans());
            trans = glm::translate(trans, m.get_trans());
//            trans = glm::translate(trans, glm::vec3(+2.7, -1.0, 0)+m.get_trans());
//            trans = glm::scale(trans, glm::vec3(6.0f));
        }
        glUniformMatrix4fv(modelShader["model"], 1, GL_FALSE, glm::value_ptr(trans));

        handTexture.use(modelShader["texHand"]);

        if (changeLights) {
            lightdir = glm::vec4(glm::sphericalRand(1.0f), 0.0f);
            lightdir2 = glm::vec4(glm::sphericalRand(1.0f), 0.0f);

//            lightcol = glm::linearRand(glm::zero<glm::vec3>(), glm::one<glm::vec3>());
//            lightcol2 = glm::linearRand(glm::zero<glm::vec3>(), glm::one<glm::vec3>());
            lightcol = glm::vec3(1.0f, 0.7f, 0.5f)*glm::linearRand(glm::one<glm::vec3>()*0.9f,
                                                                   glm::one<glm::vec3>()*1.1f);
            lightcol2 = glm::vec3(0.5f, 0.7f, 1.0f)*0.3f*glm::linearRand(glm::one<glm::vec3>()*0.9f,
                                                                         glm::one<glm::vec3>()*1.1f);
            changeLights = false;
        }

        glUniform4fv(modelShader["lightdir"], 1, glm::value_ptr(lightdir));
        glUniform3fv(modelShader["lightcol"], 1, glm::value_ptr(lightcol));

        glUniform4fv(modelShader["lightdir2"], 1, glm::value_ptr(lightdir2));
        glUniform3fv(modelShader["lightcol2"], 1, glm::value_ptr(lightcol2));

        model->draw();
        if (PLAYBACK.length() > 0) {
            // don't write on playback
        } else {
//            writer->write_poses(xt::concatenate(xt::xtuple(m.get_pose(), m.get_trans().x, m.get_trans().y, m.get_trans().z)));
//            writer->write_poses(xt::xarray<double>{m.get_trans().x, m.get_trans().y, m.get_trans().z});
            writer->write_poses(m.get_pose());
            writer->finalize_frame();
        }

        if (JOINT_OUTPUT.length() > 0) {
            xt::xarray<double> jtr = m.get_joints();
            for (int i = 0; i < jtr.shape(0); ++i) {
                for (int j = 0; j < jtr.shape(1); ++j) {
                    fprintf(jointFile, "%f ", jtr.at(i, j));
                }
            }
            fprintf(jointFile, "\n");
        }

        if (showDebugArrow) {
            auxShader.use();
            trans = glm::mat4(1.0f);
    //        trans = glm::translate(trans, glm::vec3(-0.45f, -0.05f, -0.1f));
            trans = glm::translate(trans, m.get_hand_position());
            glm::vec3 rot = m.get_hand_rotation();
            trans = glm::rotate(trans, glm::length(rot), glm::normalize(rot));
            trans = glm::rotate(trans, glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0));
    //        trans = glm::scale(trans, glm::vec3(0.15, 0.15, 0.15));
            trans = glm::scale(trans, glm::vec3(0.02, 0.02, 0.02));
//            trans = glm::scale(trans, glm::vec3(0.01, 0.01, 0.01));
    //        trans = m.transform * trans;
    //        trans = glm::translate(trans, glm::vec3(-0.45f, -0.05f, 0.4f));
            glUniformMatrix4fv(auxShader["proj"], 1, GL_FALSE, glm::value_ptr(proj));
            glUniformMatrix4fv(auxShader["view"], 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(auxShader["model"], 1, GL_FALSE, glm::value_ptr(trans));

            glUniform4fv(auxShader["lightdir"], 1, glm::value_ptr(lightdir));
            glUniform3fv(auxShader["lightcol"], 1, glm::value_ptr(lightcol));

            glUniform4fv(auxShader["lightdir2"], 1, glm::value_ptr(lightdir2));
            glUniform3fv(auxShader["lightcol2"], 1, glm::value_ptr(lightcol2));
            arrow->use();
    //        glDisable(GL_DEPTH_TEST);
            arrow->draw();
        }

        // post-process the backbuffer
        cuda_draw(&evcData, screenRender, C);

        if (screenRender) {
            // draw the quad on the screen
            Framebuffer::use_screen_fb();
            glDisable(GL_DEPTH_TEST);
            quad->use();
            quadShader.use();
            glActiveTexture(GL_TEXTURE0);
            if (showCuda) {
                glBindTexture(GL_TEXTURE_2D, evcData.cudaTexResult);
            } else {
                glBindTexture(GL_TEXTURE_2D, fb.texColorBuffer);
            }
            glUniform1i(quadShader["texFramebuffer"], 0); // texture unit 0
            glUniform1i(quadShader["doGammaCorr"], doGammaCorr && !showCuda);
            glUniform1i(quadShader["invertY"], 0); // don't invert y
            glUniform2f(quadShader["offset"], 0.0f, 0.0f);
            glUniform2f(quadShader["scale"], 1.0f, 1.0f);

            quad->draw();

            if (VIDEO_RENDER) {
                glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, videoRenderFrame);
//                std::cerr << glGetError() << std::endl;
                fwrite_unlocked(videoRenderFrame, sizeof(uint8_t), WINDOW_HEIGHT*WINDOW_WIDTH*3, stdout);
            }

            SDL_GL_SwapWindow(window);
        }

        cntr++;
        if (cntr % 100 == 0) {
            std::cerr << std::fixed << std::setprecision(2);
            std::cerr << "time=" << now << "ms" << '\t';
            std::cerr << "frames=" << cntr << '\t';
            std::cerr << "fps=" << cntr * 1000. / now << '\t';
            std::cerr << "c=" << C << '\t';
            std::cerr << "t=" << m.t << '\t';
            std::cerr << "1/dt=" << 1./m.dt << '\t';
            std::cerr << std::endl;
        }
//        if (cntr == 200) {
//            break;
//        }
    }
    if (jointFile != nullptr) {
        fflush(jointFile);
        fclose(jointFile);
    }
    destroy_evc(&evcData);
}

void parse_flags(int argc, char *argv[]) {
    using namespace clipp;
    bool showHelp = false;
    auto cli = (
        opt_value("experiment", EXPERIMENT) % "experiment output file name prefix",
        (option("-w", "--width") & value("width", WINDOW_WIDTH)) % "width",
        (option("-h", "--height") & value("height", WINDOW_HEIGHT)) % "height",
        (option("-p", "--play") & value("playback source", PLAYBACK)) % "play a recording (prepend with fifo:// for live playback)",
        option("-o", "--offscreen").set(OFFSCREEN_RENDER, true).doc("start with offscreen rendering"),
        option("-v", "--videorender").set(VIDEO_RENDER, true).doc("pipe raw video frames to stdout"),
        (option("-r", "--framerate") & value("frame rate", DISPLAY_FPS)) % "display fps",
        (option("-j", "--joints") & value("joints output filename", JOINT_OUTPUT)) % "write joints to file",
        option("--help").set(showHelp, true).doc("show this help")
    );

    if (!parse(argc, argv, cli) || showHelp) {
        std::cerr << make_man_page(cli, argv[0]);
        exit(showHelp?0:1);
    }

    const std::string fifoprot = "fifo://";
    if (PLAYBACK.length() >= fifoprot.length() && PLAYBACK.substr(0, fifoprot.length()) == fifoprot) {
        FIFO = true;
        PLAYBACK = PLAYBACK.substr(fifoprot.length());
        assert(PLAYBACK.length() > 0);
    }
}

int main(int argc, char *argv[]) {
    if (SDL_Init(SDL_INIT_EVERYTHING & (~SDL_INIT_HAPTIC)) < 0) {
        std::cerr << "Couldn't initialize SDL2: " << SDL_GetError() << std::endl;
        return 1;
    }

    parse_flags(argc, argv);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("main",
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          WINDOW_WIDTH, WINDOW_HEIGHT,
                                          SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    SDL_CHECK_NULL(window, "SDL_CreateWindow failed");

    SDL_GLContext ctx = SDL_GL_CreateContext(window);
    SDL_CHECK_NULL(ctx, "SDL_GL_CreateContext failed");

    glewExperimental = GL_TRUE;
    glewInit();

    run(window, EXPERIMENT.c_str());

    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
