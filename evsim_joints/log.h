#ifndef LOG_H
#define LOG_H

#include <SDL2/SDL.h>

#define CHECK_NULL(x, y) {if (!(x)) { SDL_Log(#x " is NULL, " y); exit(1); }}
#define CHECK_NON_NULL(x, y) {if ((x)) { SDL_Log(#x " is %d, " y, (int)x); exit(1); }}
#define SDL_CHECK_NULL(x, y) {if (!(x)) { SDL_Log(#x " is NULL, " y ": %s", SDL_GetError()); exit(1); }}

#endif // LOG_H
