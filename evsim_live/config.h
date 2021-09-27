#ifndef CONFIG_H
#define CONFIG_H

#include <string>

extern std::string EXPERIMENT;

extern std::string PLAYBACK;
extern bool FIFO;

extern int WINDOW_WIDTH;
extern int WINDOW_HEIGHT;

extern bool OFFSCREEN_RENDER;
extern bool VIDEO_RENDER;

extern long double DISPLAY_FPS;

extern std::string JOINT_OUTPUT;

#endif // CONFIG_H
