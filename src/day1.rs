use aoc_runner_derive::aoc;
use std::cmp::Reverse;
use std::collections::{HashMap, BinaryHeap};
use std::simd::{SupportedLaneCount, Simd, LaneCount, cmp::SimdPartialEq, num::SimdUint, ptr::{SimdConstPtr, SimdMutPtr}};
use std::mem::MaybeUninit;

const LINES: usize = 1000;

#[aoc(day1, part1)]
pub fn part1(input: &str) -> u64 {
    let mut left_queue = [MaybeUninit::uninit(); 1000];
    let mut right_queue = [MaybeUninit::uninit(); 1000];

    unsafe {
        for i in 0..15 {
            store_simd(parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64))).cast::<u64>().read_unaligned() }))), left_queue.as_mut_ptr().wrapping_add(i * 64));
            store_simd(parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64)) + 8).cast::<u64>().read_unaligned() }))), right_queue.as_mut_ptr().wrapping_add(i * 64));
        }

        store_simd(parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440).cast::<u64>().read_unaligned() }))), left_queue.as_mut_ptr().wrapping_add(15 * 64));
        store_simd(parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 8).cast::<u64>().read_unaligned() }))), right_queue.as_mut_ptr().wrapping_add(15 * 64));


        store_simd(parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448).cast::<u64>().read_unaligned() }))), left_queue.as_mut_ptr().wrapping_add(15 * 64 + 32));
        store_simd(parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448 + 8).cast::<u64>().read_unaligned() }))), right_queue.as_mut_ptr().wrapping_add(15 * 64 + 32));

        let mut left_queue = MaybeUninit::array_assume_init(left_queue);
        let mut right_queue = MaybeUninit::array_assume_init(right_queue);
        left_queue.sort_unstable();
        right_queue.sort_unstable();
        let mut distance = 0;
        for i in 0..LINES {
            let left = left_queue[i];
            let right = right_queue[i];
            distance += left.abs_diff(right);
        }

        debug_assert!(left_queue.is_empty());
        debug_assert!(right_queue.is_empty());

        distance
    }
}

#[aoc(day1, part2)]
pub fn part2(input: &str) -> u64 {
    let mut sum = 0;
    let mut bitmask = Bitmask { bitmask: [Simd::splat(0); 391] };

    for i in 0..15 {
        bitmask.set_many(parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64))).cast::<u64>().read_unaligned() }))));
    }

    bitmask.set_many(parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440).cast::<u64>().read_unaligned() }))));
    bitmask.set_many(parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448).cast::<u64>().read_unaligned() }))));


    for i in 0..15 {
        let value = bitmask.filter_many(parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64)) + 8).cast::<u64>().read_unaligned() }))));
        sum += value.reduce_sum();
    }

    let value = bitmask.filter_many(parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 8).cast::<u64>().read_unaligned() }))));
    sum += value.reduce_sum();

    let value = bitmask.filter_many(parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448 + 8).cast::<u64>().read_unaligned() }))));
    sum += value.reduce_sum();
    sum
}

#[inline(always)]
fn parse_many_5_digit_numbers<const N: usize>(input: Simd<u64, N>) -> Simd<u64, N> where LaneCount<N>: SupportedLaneCount {
    // This will use SWAR to parse a number and SIMD to perform multiple SWAR operations at the
    // same time.

    // From each number, only include the first 5 digits, which will be referred to using n0 to n4. With a number like 12345, n0 is 1, n2 is 2, etc.
    let value = input & (Simd::splat(0xFF_FF_FF_FF_FF));

    // Make each n be from 0..9 instead of from ASCII 0..9
    let value = value - Simd::splat(u64::from_le_bytes(*b"00000\0\0\0"));

    // Combine pairs of n so that n0 = n0 * 10 + n1, n2 = n2 * 10 + n3, etc.
    // This is done using (value * 10) + (value >> 8) in the below optimized form
    let value = (value * Simd::splat((10 << 8) + 1)) >> Simd::splat(8);

    // The value of odd-numbered n are ignored.
    let value = value & Simd::splat(0xFF_00_FF_00_FF);

    // Combine pairs of u16s (e.g. m0 = (n0, n1), m1 = (n2, n3), etc)) so that m0 = m0 * 100 + m1. The value of odd-numbered u16s are ignored.
    // This is done using (value * 100) + (value >> 16) in the below optimized form
    let value = (value * Simd::splat((100 << 16) + 1)) >> Simd::splat(16);

    // Keep the lower 16 bits and add n4 to it. n4 can be obtained from m2. m2 has already been multiplied by 1000, so m4 also needs to divide it by 1000.
    // To get n4, this usees (value >> 32) / 1000, which can be optimized into ((value >> 32) * 274877907) >> 38
    ((value & Simd::splat(0xFFFF)) * Simd::splat(10)) + (((value >> Simd::splat(32)) * Simd::splat(274877907)) >> Simd::splat(38))
}

#[inline(always)]
unsafe fn store_simd<const N: usize>(source: Simd<u64, N>, dest: *mut MaybeUninit<u64>) where LaneCount<N>: SupportedLaneCount {
    unsafe { source.scatter_ptr(Simd::splat(dest.cast::<u64>()).wrapping_add(Simd::from_array(std::array::from_fn(|i| i)))); }
}

struct Bitmask {
    pub bitmask: [Simd<u64, 4>; 391],
}

impl Bitmask {
    #[inline(always)]
    fn set_many<const N: usize>(&mut self, pos: Simd<u64, N>) where LaneCount<N>: SupportedLaneCount {
        let pos: Simd::<usize, N> = unsafe { std::intrinsics::transmute_unchecked(pos) };
        let byte_pos = Simd::splat(self.bitmask.as_mut_ptr().cast::<u8>()).wrapping_add(pos >> Simd::splat(3));
        unsafe { (Simd::gather_ptr(byte_pos.cast_const()) | (Simd::splat(1) << (Simd::from_array(pos.to_array().map(|x| x as u8)) & Simd::splat(7)))).scatter_ptr(byte_pos) };
    }

    #[inline(always)]
    fn filter_many<const N: usize>(&self, data: Simd<u64, N>) -> Simd<u64, N> where LaneCount<N>: SupportedLaneCount {
        let data_usize: Simd::<usize, N> = unsafe { std::intrinsics::transmute_unchecked(data) };
        let bytes = unsafe { Simd::gather_ptr(Simd::splat(self.bitmask.as_ptr().cast::<u8>()).wrapping_add(data_usize >> Simd::splat(3))) };
        let bit_masks = Simd::splat(1) << (Simd::from_array(data.to_array().map(|x| x as u8)) & Simd::splat(7));
        Simd::from_array((bytes & bit_masks).to_array().map(|x| x as u64)).simd_ne(Simd::splat(0u64)).select(data, Simd::splat(0u64))
    }
}
