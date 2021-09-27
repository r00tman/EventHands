#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    uint16_t x;
    uint8_t y;
    uint8_t p;
} event_t;

#define NUM (512*512*64)

event_t buf[NUM];

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        puts("need a fn");
    }

    FILE *f = fopen(argv[1], "rb");

    uint64_t buf_size = fread_unlocked(buf, sizeof(event_t), NUM, f);
    uint64_t off = 0;

    uint64_t cnt = 0;
    for (int i = 0; i < 10000000;) {
        if (buf[cnt-off].p == 255) {
            ++i;
        }
        ++cnt;
        if (cnt-off >= buf_size) {
            off += buf_size;
            buf_size = fread_unlocked(buf, sizeof(event_t), NUM, f);
        }
    }
    fclose(f);
    printf("%llu\n", cnt);
    return 0;
}
