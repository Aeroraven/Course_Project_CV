#pragma once

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <cstdio>

#include <opencv2/opencv.hpp>

#define f64 double
#define f32 float
#define i32 int
#define i64 long long
#define u32 unsigned int
#define u64 unsigned long long
#define u8 unsigned char

#define Min(a,b) ((a)<(b)?(a):(b))
#define Max(a,b) ((a)>(b)?(a):(b))

namespace Common {
	struct Vec2 {
		f64 x, y;
	};
}
