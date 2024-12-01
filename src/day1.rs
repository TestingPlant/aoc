use aoc_runner_derive::aoc;
use std::cmp::Reverse;
use std::collections::{HashMap, BinaryHeap};

const LINES: usize = 1000;
#[aoc(day1, part1)]
pub fn part1(input: &str) -> u32 {
    let mut left_queue = BinaryHeap::with_capacity(LINES);
    let mut right_queue = BinaryHeap::with_capacity(LINES);

    for line in input.lines() {
        let left: u32 = (&line[0..5]).parse().unwrap();
        let right: u32 = (&line[8..13]).parse().unwrap();
        left_queue.push(Reverse(left));
        right_queue.push(Reverse(right));
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
