use aoc_runner_derive::aoc;
use std::simd::{SupportedLaneCount, Simd, Mask, LaneCount, cmp::{SimdPartialEq, SimdOrd, SimdPartialOrd}, num::SimdUint, ptr::{SimdConstPtr, SimdMutPtr}};
use std::mem::MaybeUninit;

const LINES: usize = 1000;

#[aoc(day2, part1)]
pub fn part1(input: &str) -> u64 {
    let mut sum = 0;
    let mut start = input.as_bytes().as_ptr();
    unsafe {
        for i in 0..15 {
            let mut data: Simd<u64, 64> = Simd::splat(0);
            let mut len: Simd<u64, 64> = Simd::splat(0);
            for j in 0..64 {
                (data[j], len[j]) = read_to_swar(&mut start);
            }
            sum += count_safe(data, len);
        }

        let mut data: Simd<u64, 32> = Simd::splat(0);
        let mut len: Simd<u64, 32> = Simd::splat(0);
        for j in 0..32 {
            (data[j], len[j]) = read_to_swar(&mut start);
        }
        sum += count_safe(data, len);

        let mut data: Simd<u64, 8> = Simd::splat(0);
        let mut len: Simd<u64, 8> = Simd::splat(0);
        for j in 0..7 {
            (data[j], len[j]) = read_to_swar(&mut start);
        }
        (data[7], len[7]) = read_to_swar_maybe_null_terminated(&mut start);

        sum
    }
}

#[aoc(day2, part2)]
pub fn part2(input: &str) -> u64 {
    let mut sum = 0;
    let mut start = input.as_bytes().as_ptr();
    unsafe {
        for i in 0..15 {
            let mut data: Simd<u64, 64> = Simd::splat(0);
            let mut len: Simd<u64, 64> = Simd::splat(0);
            for j in 0..64 {
                (data[j], len[j]) = read_to_swar(&mut start);
            }
            sum += count_safe_part_2(data, len);
        }

        let mut data: Simd<u64, 32> = Simd::splat(0);
        let mut len: Simd<u64, 32> = Simd::splat(0);
        for j in 0..32 {
            (data[j], len[j]) = read_to_swar(&mut start);
        }
        sum += count_safe_part_2(data, len);

        let mut data: Simd<u64, 8> = Simd::splat(0);
        let mut len: Simd<u64, 8> = Simd::splat(0);
        for j in 0..7 {
            (data[j], len[j]) = read_to_swar(&mut start);
        }
        (data[7], len[7]) = read_to_swar_maybe_null_terminated(&mut start);
        sum += count_safe_part_2(data, len);

        sum
    }
}

// Reads line into swar-encoded numbers and length
unsafe fn read_to_swar(line_ptr: &mut *const u8) -> (u64, u64) {
    let line = line_ptr.cast::<Simd<u8, 32>>().read_unaligned();
    let line_len = line.simd_eq(Simd::splat(b'\n')).to_bitmask().trailing_zeros() as usize;
    let line_delims = line.simd_lt(Simd::splat(b'0'));

    const ZERO: u64 = b'0' as u64;
    let mut result = 0u64;
    let mut len = 0u64;
    let mut i = 0usize;
    while i < line_len {
        let first_digit = line[i] as u64 - ZERO;
        if line_delims.test_unchecked(i + 1) {
            // parse 1 digit
            result |= first_digit << len;
            i += 2;
        } else {
            // parse 2 digits
            result |= ((first_digit * 10) + ((line[i + 1] as u64 - ZERO))) << len;
            i += 3;
        }
        len += 8;
    }

    *line_ptr = line_ptr.add(i);

    (result, len)
}   

// Reads line into swar-encoded numbers and length
unsafe fn read_to_swar_maybe_null_terminated(line_ptr: &mut *const u8) -> (u64, u64) {
    let line = line_ptr.cast::<Simd<u8, 32>>().read_unaligned();
    let line_len = (line.simd_eq(Simd::splat(b'\n')) | line.simd_eq(Simd::splat(0))).to_bitmask().trailing_zeros() as usize;
    let line_delims = line.simd_lt(Simd::splat(b'0'));

    const ZERO: u64 = b'0' as u64;
    let mut result = 0u64;
    let mut len = 0u64;
    let mut i = 0usize;
    while i < line_len {
        let first_digit = line[i] as u64 - ZERO;
        if line_delims.test_unchecked(i + 1) {
            // parse 1 digit
            result |= first_digit << len;
            i += 2;
        } else {
            // parse 2 digits
            result |= ((first_digit * 10) + ((line[i + 1] as u64 - ZERO))) << len;
            i += 3;
        }
        len += 8;
    }

    *line_ptr = line_ptr.add(i);

    (result, len)
}   

#[inline]
fn is_safe<const N: usize>(data: Simd<u64, N>, len: Simd<u64, N>) -> Mask<i64, N> where LaneCount<N>: SupportedLaneCount {
    // Mask for the numbers stored in SWAR without the padding
    let len_mask = Simd::splat(u64::MAX) >> (Simd::splat(64) - len);

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

    monotonic & at_most_3 & no_duplicates
}

#[inline(always)]
fn count_safe<const N: usize>(data: Simd<u64, N>, len: Simd<u64, N>) -> u64 where LaneCount<N>: SupportedLaneCount {
    count_ones(is_safe(data, len))
}

#[inline(always)]
fn count_safe_part_2<const N: usize>(data: Simd<u64, N>, len: Simd<u64, N>) -> u64 where LaneCount<N>: SupportedLaneCount {
    let len_sub_8 = len - Simd::splat(8);
    // Assume len >= (5*8)
    count_ones(
        is_safe(data, len) |
        is_safe(remove::<0, N>(data), len_sub_8) |
        is_safe(remove::<1, N>(data), len_sub_8) |
        is_safe(remove::<2, N>(data), len_sub_8) |
        is_safe(remove::<3, N>(data), len_sub_8) |
        is_safe(remove::<4, N>(data), len_sub_8) |
        (is_safe(remove::<5, N>(data), len_sub_8) & (len.simd_ge(Simd::splat(48)))) |
        (is_safe(remove::<6, N>(data), len_sub_8) & (len.simd_ge(Simd::splat(56)))) |
        (is_safe(remove::<7, N>(data), len_sub_8) & (len.simd_eq(Simd::splat(64))))
    )
}

#[inline(always)]
fn count_ones<const N: usize>(mask: Mask<i64, N>) -> u64 where LaneCount<N>: SupportedLaneCount {
    mask.select(Simd::splat(1u64), Simd::splat(0u64)).reduce_sum()
}

#[inline(always)]
fn remove<const I: u64, const N: usize>(value: Simd<u64, N>) -> Simd<u64, N> where LaneCount<N>: SupportedLaneCount {
    let lower_mask = (1u64 << (I*8)) - 1;
    let upper_mask = !lower_mask << 8;
    ((value & Simd::splat(upper_mask)) >> Simd::splat(8)) | (value & Simd::splat(lower_mask))
}
