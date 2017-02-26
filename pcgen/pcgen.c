#include <stdio.h>
#include <math.h>

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

float thresh_min;
float thresh_max;

EXPORT void set_bounds(float min, float max) {
    thresh_min = min;
    thresh_max = max;
}

int use_pt(float depth) {
    return (depth > thresh_min) && (depth < thresh_max);
}

EXPORT int get_num_points(float * data, int data_len) {
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

EXPORT void build_pc(float * data, int data_len, float * output, float d, int w, int h, float zcale) {
    int pt_idx = 0;
    int u, v;
    float depth;

    vec3 vpt;
    vec3 r;

    vec3 * out_pts = (vec3 *) output;

    vec3 origin;
    origin.x = 0;
    origin.y = 0;
    origin.z = d; 

    for (u = 0; u < w; u++) {
        for (v = 0; v < h; v++) {
            depth = data[v * w + u];

            if (use_pt(depth)) {                
                vpt.x = ((float)u - (float)w/2.0) / (float)w;
                vpt.y = ((float)v - (float)h/2.0) / (float)w;
                vpt.z = 0;

                sub(&r, &vpt, &origin);
                normalize(&r);
                scale(&r, depth);
                add(&out_pts[pt_idx], &vpt, &r);

                out_pts[pt_idx].z *= zcale;

                pt_idx++;
            }
        }
    }
}