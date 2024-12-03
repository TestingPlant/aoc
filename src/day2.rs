use aoc_runner_derive::aoc;
use std::simd::{SupportedLaneCount, Simd, LaneCount, cmp::{SimdPartialEq, SimdOrd}, num::SimdUint, ptr::{SimdConstPtr, SimdMutPtr}};
use std::mem::MaybeUninit;

const LINES: usize = 1000;

#[aoc(day2, part1)]
pub fn part1(input: &str) -> u64 {
    let mut sum = 0;
    let mut lines = input.lines();
    unsafe {
        for i in 0..15 {
            let mut data: Simd<u64, 64> = Simd::splat(0);
            let mut len: Simd<u64, 64> = Simd::splat(0);
            for j in 0..64 {
                let (line_data, line_len) = read_to_swar(lines.next().unwrap());
                data[j] = line_data;
                len[j] = line_len;
            }
            sum += count_safe(data, len);
        }

        let mut data: Simd<u64, 32> = Simd::splat(0);
        let mut len: Simd<u64, 32> = Simd::splat(0);
        for j in 0..32 {
            let (line_data, line_len) = read_to_swar(lines.next().unwrap());
            data[j] = line_data;
            len[j] = line_len;
        }
        sum += count_safe(data, len);

        let mut data: Simd<u64, 8> = Simd::splat(0);
        let mut len: Simd<u64, 8> = Simd::splat(0);
        for j in 0..8 {
            let (line_data, line_len) = read_to_swar(lines.next().unwrap());
            data[j] = line_data;
            len[j] = line_len;
        }
        sum += count_safe(data, len);

        sum
    }
}

// Reads line into swar-encoded numbers and length
fn read_to_swar(line: &str) -> (u64, u64) {
    let mut result = 0;
    let mut len = 0;
    for (i, n) in line.split(' ').enumerate() {
        result |= n.parse::<u64>().unwrap() << (i * 8);
        len += 1;
    }
    (result, len)
}

#[inline(always)]
fn count_safe<const N: usize>(data: Simd<u64, N>, len: Simd<u64, N>) -> u64 where LaneCount<N>: SupportedLaneCount {
    // Mask for the numbers stored in SWAR without the padding
    let len_mask = Simd::splat(u64::MAX) >> (Simd::splat(64) - (len * Simd::splat(8)));

    // let increasing_mask = (data > (data >> 8));
    // Find the difference between pairs of numbers. Each difference is an i8. The most significant 8 bytes should be
    // ignored.
    let difference = data - (data >> Simd::splat(8));
    let difference_mask = len_mask >> Simd::splat(8);

    const SIGN_BITS: u64 = 0x8080808080808080;
    let sign_bits_mask = Simd::splat(SIGN_BITS) & difference_mask;
    let sign_bits = difference & sign_bits_mask;
    // is_positive assumes that the differences are monotonic
    let is_positive = sign_bits.simd_eq(Simd::splat(0));

    // Check that the sign bit is either set for the entire len or that the number is positive
    let monotonic = sign_bits.simd_eq(sign_bits_mask) | is_positive;

    let twos_complement_difference = !difference + Simd::splat(1);
    let abs_difference = is_positive.select(difference, twos_complement_difference) & (len_mask >> Simd::splat(8));

    let at_most_3 = abs_difference.simd_eq(abs_difference & Simd::splat(0x0303030303030303));

    // If the differences in data_adjusted are not monotonic, there are duplicate numbers and is unsafe.
    let adjusted_data = data + is_positive.select(Simd::splat(0x0706050403020100), Simd::splat(0x0001020304050607)) & len_mask;
    let adjusted_difference = adjusted_data - (adjusted_data >> Simd::splat(8));
    let adjusted_twos_complement_difference = !adjusted_difference + Simd::splat(1);
    let adjusted_abs_difference = is_positive.select(adjusted_difference, adjusted_twos_complement_difference);
    // Check if the differences are monotonic. This is different from the above calculation because
    // consecutive values in the adjusted data are safe, leading to differences of zero.
    let no_duplicates = (adjusted_abs_difference & sign_bits_mask).simd_eq(Simd::splat(0));

    let safe = monotonic & at_most_3 & no_duplicates;
    safe.select(Simd::splat(1u64), Simd::splat(0u64)).reduce_sum()
}
