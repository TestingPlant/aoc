use aoc_runner_derive::aoc;
use std::cmp::Reverse;
use std::collections::{HashMap, BinaryHeap};
use std::simd::{SupportedLaneCount, Simd, LaneCount};

const LINES: usize = 1000;

#[aoc(day1, part1)]
pub fn part1(input: &str) -> u32 {
    let mut left_queue = BinaryHeap::with_capacity(LINES);
    let mut right_queue = BinaryHeap::with_capacity(LINES);

    for i in 0..15 {
        for n in parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64))).cast::<u64>().read_unaligned() }))).to_array() {
            left_queue.push(Reverse(n as u32));
        }
        for n in parse_many_5_digit_numbers(Simd::<u64, 64>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + (i * (14*64)) + 8).cast::<u64>().read_unaligned() }))).to_array() {
            right_queue.push(Reverse(n as u32));
        }
    }

    for n in parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440).cast::<u64>().read_unaligned() }))).to_array() {
        left_queue.push(Reverse(n as u32));
    }
    for n in parse_many_5_digit_numbers(Simd::<u64, 32>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 8).cast::<u64>().read_unaligned() }))).to_array() {
        right_queue.push(Reverse(n as u32));
    }

    for n in parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448).cast::<u64>().read_unaligned() }))).to_array() {
        left_queue.push(Reverse(n as u32));
    }
    for n in parse_many_5_digit_numbers(Simd::<u64, 8>::from_array(std::array::from_fn(|j| unsafe { input.as_ptr().add((j * 14) + 13440 + 448 + 8).cast::<u64>().read_unaligned() }))).to_array() {
        right_queue.push(Reverse(n as u32));
    }

    let mut distance = 0;
    for _ in 0..LINES {
        let left = left_queue.pop().unwrap().0;
        let right = right_queue.pop().unwrap().0;
        distance += left.abs_diff(right);
    }

    debug_assert!(left_queue.is_empty());
    debug_assert!(right_queue.is_empty());

    distance
}

#[aoc(day1, part2)]
pub fn part2(input: &str) -> u32 {
    let mut map = HashMap::with_capacity(LINES * 2);

    const PRESENT_FLAG: u32 = 1 << 31;

    for line in input.lines() {
        let left: u32 = (&line[0..5]).parse().unwrap();
        let right: u32 = (&line[8..13]).parse().unwrap();

        map.entry(left).and_modify(|x| *x |= PRESENT_FLAG).or_insert(PRESENT_FLAG);
        map.entry(right).and_modify(|x| *x += right).or_insert(right);
    }

    map.into_iter().map(|(_, x)| x).filter(|x| x & PRESENT_FLAG != 0).map(|x| x & !PRESENT_FLAG).sum()
}

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
