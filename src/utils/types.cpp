#include "types.hpp"

#include <cstring>

namespace llaisys::utils {
float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));

    uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000);
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t mant = f32 & 0x7FFFFF;

    if (exp == 0xFF) {
        if (mant == 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
        }
        uint16_t nan_payload = static_cast<uint16_t>(mant >> 13);
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00 | nan_payload | 1)};
    }

    int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;
    if (half_exp >= 31) {
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    }

    if (half_exp <= 0) {
        if (half_exp < -10) {
            return fp16_t{sign};
        }
        mant |= 0x800000;
        uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        uint32_t half_mant = mant >> shift;
        uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) {
            half_mant++;
        }
        return fp16_t{static_cast<uint16_t>(sign | half_mant)};
    }

    uint16_t half =
        static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exp) << 10) | static_cast<uint16_t>(mant >> 13));
    uint32_t round_bits = mant & 0x1FFF;
    if (round_bits > 0x1000 || (round_bits == 0x1000 && (half & 1u))) {
        half++;
    }
    return fp16_t{half};
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF +
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
} // namespace llaisys::utils
