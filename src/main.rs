use std::{cmp::Ordering, collections::HashMap, f32, ops::ControlFlow};

use anyhow::{Context, Result};

const ODDS_DATA: &str = include_str!("../nfl_odds.csv");

fn read_csv(s: &str) -> Result<(Vec<&str>, Vec<f32>)> {
    // using split instaed of split lines to avoid creating Strings
    let mut iter = s
        .split("\r\n")
        .flat_map(|s| s.split("\n"))
        .filter(|s| !s.is_empty());

    // I simply do not care about dealing with quote chars atm
    let teams: Vec<&str> = iter
        .next()
        .context("unable to parse first line")?
        .split(',')
        .map(str::trim)
        .collect();

    // check for the right number of probabilities in each row
    let num_teams = teams.len();
    let num_rounds = iter.try_fold(0usize, |acc, line| {
        (line.split(",").take(num_teams + 1).count() == num_teams)
            .then_some(acc + 1)
            .context("line has wrong number of entries")
    })?;

    let odds: Vec<f32> = match s
        .split("\r\n")
        .flat_map(|s| s.split("\n"))
        .filter(|s| !s.is_empty())
        .skip(1)
        .flat_map(|line| line.split(',').map(str::trim))
        .map(|s| {
            if s.is_empty() {
                Ok(f32::NAN)
            } else {
                s.parse::<f32>()
                    .with_context(|| format!("unable to parse {:?} as a f32", s))
            }
        })
        .try_fold(
            Vec::with_capacity(num_rounds * num_teams),
            |mut acc, x| match x {
                Ok(val) => {
                    acc.push(val);
                    ControlFlow::Continue(acc)
                }
                Err(err) => ControlFlow::Break(err),
            },
        ) {
        ControlFlow::Break(err) => return Err(err),
        ControlFlow::Continue(res) => res,
    };

    Ok((teams, odds))
}

type TeamsMask = u32;

const fn bin_co(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    let mut num: u64 = 1;
    let mut i: u32 = 1;
    while i <= k {
        num *= (n + 1 - i) as u64;
        // this division won't lose any perscision as num will have mulitplied at least `i`
        // consecutive nums up to this pooint and thus have a factor of n in it
        num /= i as u64;

        i += 1;
    }

    num as u32
}

const SSTEP_N: u32 = 5;
/// smoothstep function change `SSTEP_N` to affect how smooth it is
fn sstep(x: f32) -> f32 {
    const COEFFS: [f32; SSTEP_N as usize + 1] = {
        let mut arr = [0f32; SSTEP_N as usize + 1];

        let mut idx: u32 = 0;
        while (idx as usize) < arr.len() {
            let fst = bin_co(SSTEP_N + idx, idx);
            let snd = bin_co(2 * SSTEP_N + 1, SSTEP_N - idx);
            let coeff =
                ((fst as f32) * (snd as f32)).copysign(if idx % 2 == 1 { -1.0 } else { 1.0 });

            arr[idx as usize] = coeff;

            idx += 1;
        }

        arr
    };
    if x.is_nan() {
        x
    } else if x >= 1.0 {
        1.0
    } else if x <= 0.0 {
        0.0
    } else {
        COEFFS
            .iter()
            .copied()
            .enumerate()
            .map(|(k, coeff)| (x.powi((SSTEP_N + k as u32 + 1) as i32) * coeff) as f64)
            .sum::<f64>() as f32
    }
}

fn main() -> Result<()> {
    let (teams, odds) = read_csv(ODDS_DATA).context("unable to read the csv file")?;
    (TeamsMask::BITS as usize >= teams.len())
        .then_some(())
        .with_context(|| {
            format!(
                "too many teams for the size of `TeamsMask`: {}",
                teams.len()
            )
        })?;

    // println!("teams: {:?}\n\nodds: {:?}", teams, odds);

    let mut search_stack = vec![(0 as TeamsMask, 2u32); 1];
    let num_teams = teams.len();

    let max_round = 5u32;
    let max_strikes = 2u32;

    (max_round as usize <= odds.len() / teams.len())
        .then_some(())
        .context("you must have data for at least `max_rounds` rounds")?;

    // this is your assumed probability for winning if you exit in round n
    let winning_prob: Vec<_> = (0..=max_round)
        .map(|x| {
            x.saturating_sub(2.min(max_round.saturating_sub(1))) as f32
                / max_round.saturating_sub(2.min(max_round.saturating_sub(1))) as f32
        })
        .map(sstep)
        .collect();

    // the maximum expected win probability given teams already selected and the number of strikes
    // remaining
    let mut max_expected: HashMap<(TeamsMask, u32), f32> = HashMap::new();

    // probabilities of every team winning or tying in the `n`th round is given by
    // `&prob[num_teams * n..num_teams * (n + 1)` int the form of tuples of `(team, prob)` where
    // `team` is the index into the `teams` vector, and `prob` is the probability that they win,
    // from the best team to the worst team sorted by win/tie probability. A `NaN` win/tie
    // probability corresponds to a bye week i.e. there is no game for the team and you cannot pick
    // them.
    let probs = {
        let mut curr_probs: Vec<_> = odds
            .chunks_exact(teams.len())
            .flat_map(|chunk| chunk.iter().copied().enumerate())
            .map(|(i, x)| ((1 as TeamsMask) << i as u32, x))
            .collect();
        // println!("{:#?}", curr_probs);
        curr_probs.chunks_exact_mut(teams.len()).for_each(|chunk| {
            chunk.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or_else(|| (!a.1.is_nan()).cmp(&!b.1.is_nan()))
            })
        });

        curr_probs
    };

    // let max_max_ev: Vec<_> = probs
    //     .chunks_exact(teams.len())
    //     .take(max_round as usize)
    //     .rev()
    //     .fold(vec![0.0; max_round as usize + 1], |mut acc, chunk| {
    //         let prob = chunk.last().unwrap().1;
    //         acc.push(prob);

    //         acc.reserve(max_strikes as usize);
    //         let beg = acc.len() - max_strikes as usize - 2;
    //         let end = acc.len() - 1;
    //             acc[beg..end]
    //                 .windows(2)
    //                 .map(|wind| (1.0 - prob) * wind[0] + prob * wind[1]).zip(acc.)
    //         ;
    //         acc
    //     });

    // A vector of teams to add to the search stack, kept outside of the loop, so avoid reallocation
    let mut to_add: Vec<(TeamsMask, u32)> = Vec::new();
    while !search_stack.is_empty() {
        // Safety: garunteed by the while loop check
        let (mask, strikes) = unsafe { search_stack.pop().unwrap_unchecked() };

        let round = mask.count_ones() + max_strikes - strikes;

        let curr_round = &probs[round as usize * num_teams..(round + 1) as usize * num_teams];

        if round >= max_round {
            max_expected.insert((mask, strikes), 1.0);
            continue;
        }

        to_add.extend(
            curr_round
                .iter()
                .filter(|&(i, p)| !p.is_nan() && (mask & i) == 0)
                .map(|&(i, _x)| (mask | i, strikes))
                .filter(|k| !max_expected.contains_key(k)),
        );

        if strikes > 0 {
            to_add.extend(
                curr_round
                    .iter()
                    .filter(|&(i, p)| !p.is_nan() && (mask & i) == 0)
                    .map(|&(i, _x)| (mask | i, strikes - 1))
                    .filter(|k| !max_expected.contains_key(k))
                    .peekable(),
            );
        }
        if !to_add.is_empty() {
            search_stack.push((mask, strikes));
            search_stack.append(&mut to_add);

            continue;
        }

        // TODO see about implementing any kind of pruning
        let max = if strikes > 0 {
            curr_round
                .iter()
                .rev()
                .take(max_round as usize)
                .filter(|&(i, p)| !p.is_nan() && (mask & i) == 0)
                .map(|&(team, prob)| {
                    (max_expected[&(mask | team, strikes)] + 1.0) * prob
                        + (max_expected[&(mask | team, strikes - 1)] + 1.0) * (1.0 - prob)
                })
                .max_by(|p1, p2| p1.partial_cmp(p2).unwrap())
                .unwrap_or(f32::NAN)
        } else {
            curr_round
                .iter()
                .rev()
                .take(max_round as usize)
                .filter(|&(i, p)| !p.is_nan() && (mask & i) == 0)
                .map(|&(team, prob)| (max_expected[&(mask | team, strikes)] + 1.0) * prob)
                .max_by(|p1, p2| p1.partial_cmp(p2).unwrap())
                .unwrap_or(f32::NAN)
        };

        max_expected.insert((mask, strikes), max);
    }

    let mut results: Vec<_> = max_expected
        .drain()
        .map(|((m, s), mm)| (m, s, mm))
        .collect();
    results.sort_unstable_by(|a, b| (a.0.count_ones(), b.1).cmp(&(b.0.count_ones(), a.1)));
    for (mask, s, ev) in &results[0..100] {
        println!("mask: {:>032b} strikes: {:>2} ev: {:>30}", mask, s, ev);
    }

    // probs[0..num_teams].iter().for_each(|x| {
    //     println!(
    //         "team: {:>30} max EV: {:>30}",
    //         teams[x.0.trailing_zeros() as usize],
    //         max_expected[&(x.0, 2u32, 1u32)]
    //     )
    // });

    Ok(())
}
