#include "math.h"

int use_pt(float depth) {
    return (depth > 0.2) && (depth < 0.7);
}

int get_num_points(float * data, int data_len) {
    int num_points = 0;
    int i;

    for (i = 0; i < data_len; i++) {
        if (use_pt(data[i])) {
            num_points++;
        }
    }

    return num_points;
}

typedef struct {
    float x;
    float y;
    float z;
} vec3;

void normalize(vec3 * pt) {
    float norm = sqrt(pt->x * pt->x + pt->y * pt->y + pt->z * pt->z);

    pt->x = pt->x / norm;
    pt->y = pt->y / norm;
    pt->z = pt->z / norm;
}

void sub(vec3 * dst, vec3 * p1, vec3 * p2) {
    dst->x = p1->x - p2->x;
    dst->y = p1->y - p2->y;
    dst->z = p1->z - p2->z;
}

void add(vec3 * dst, vec3 * p1, vec3 * p2) {
    dst->x = p1->x + p2->x;
    dst->y = p1->y + p2->y;
    dst->z = p1->z + p2->z;
}

void scale(vec3 * dst, float factor) {
    dst->x *= factor;
    dst->y *= factor;
    dst->z *= factor;
}

void build_pc(float * data, int data_len, float * output, float d, int w, int h) {
    int pt_idx = 0;
    int u, v;
    float depth;

    vec3 * out_pts = (vec3 *) output;

    vec3 origin;
    origin.x = 0;
    origin.y = 0;
    origin.z = d; 

    for (u = 0; u < w; u++) {
        for (v = 0; v < h; v++) {
            depth = data[v * w + h];

            if (use_pt(depth)) {
                float xp = (u - w/2) / (float)w;
                float yp = (h - h/2) / (float)w;

                vec3 vpt;
                vpt.x = xp;
                vpt.y = yp;
                vpt.z = 0;

                vec3 r;
                sub(&r, &vpt, &origin);
                normalize(&r);
                scale(&r, depth);
                add(&out_pts[pt_idx], &vpt, &r);

                pt_idx++;
            }
        }
    }
}