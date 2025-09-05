use std::{collections::HashMap, f32, fs, iter, ops::ControlFlow, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, command};

/// A simple program for derriving the max win proability for a survivor pool based on a table of
/// win probabilities for each round in a csv file.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the csv file containing the probability data
    #[arg(short, long, value_name = "FILE")]
    prob_file_path: PathBuf,

    #[arg(value_name = "TEAM")]
    teams: Vec<String>,

    /// The first round of the the probability data (not neccessarilly the current round of the
    /// pool)
    #[arg(short, long, default_value_t = 0)]
    starting_round: u32,

    /// The number of mulligans you have at your disposal
    #[arg(short, long, default_value_t = 0)]
    mulligans: u32,
}

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

const SSTEP_N: u32 = 2;
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
            .sum::<f64>()
            .clamp(0.0, 1.0) as f32
    }
}

fn solve(
    odds: &[f32],
    winning_prob: &[f32],
    num_teams: usize,
    curr_teams: TeamsMask,
    mulligans: u32,
    round_offset: u32,
) -> Result<Vec<(usize, f32, f32)>> {
    let max_round = (odds.len() / num_teams) as u32 + round_offset;
    // dbg!(max_round);
    // return Ok(());

    (odds.len() % num_teams == 0)
        .then_some(())
        .context("you must have odds for every team in every week")?;

    (curr_teams.count_ones() >= round_offset)
        .then_some(())
        .context("round offset can not be less than the number of already selected teams")?;

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
            .chunks_exact(num_teams)
            .flat_map(|chunk| chunk.iter().copied().enumerate())
            .map(|(i, x)| ((1 as TeamsMask) << i as u32, x))
            .collect();
        // println!("{:#?}", curr_probs);
        curr_probs.chunks_exact_mut(num_teams).for_each(|chunk| {
            chunk.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or_else(|| (!a.1.is_nan()).cmp(&!b.1.is_nan()))
                    .reverse()
            })
        });

        curr_probs
    };

    // a vector of the maximum theoretical win pct for a given number of teams and strikes where the
    // where the number of teams picked so far is `t` and the number of strikes remaining is `s`,
    // then `max_for_teams_strikes[t * max_strikes + s]` is an upper bound on the possible win prob
    // with this
    let max_for_teams_mulligans = {
        let mut temp = probs
            .chunks_exact(num_teams)
            .take(max_round as usize)
            .map(|chunk| chunk.first().map(|&(_i, p)| p))
            .enumerate()
            .rev()
            .try_fold(
                vec![1.0f32; mulligans as usize + 1],
                |mut acc, (i, p)| -> Result<Vec<_>> {
                    let row_sz = mulligans as usize + 1;
                    acc.extend_from_within(acc.len() - row_sz..);
                    let bisection_pt = acc.len() - row_sz;
                    let (prev_arr, new_arr) = acc.split_at_mut(bisection_pt);
                    let prev_prob = &prev_arr[prev_arr.len() - row_sz..];
                    let win_prob_for_round = winning_prob[i];

                    let p = p.context("missing probability")?;

                    new_arr
                        .iter_mut()
                        .zip(iter::once(win_prob_for_round).chain(prev_prob.iter().copied()))
                        .for_each(|(p_w, p_l)| *p_w = *p_w * p + p_l * (1.0 - p));

                    Ok(acc)
                },
            )
            .context("unable to create max array")?;

        temp.reverse();
        temp
    };

    let curr_mask = curr_teams;
    {
        // helper function for creating the iterator, had to be this way to store the result in a
        // vector bc I can't name the iterator's type
        let get_iter = |mask: TeamsMask, strikes: u32| {
            let round = mask.count_ones() as usize;
            let max_pw = max_for_teams_mulligans
                [(round - round_offset as usize + 1) * mulligans as usize + strikes as usize];
            let max_pl = if strikes > 0 {
                max_for_teams_mulligans[(round - round_offset as usize + 1) * mulligans as usize
                    + strikes as usize
                    - 1]
            } else {
                winning_prob[round]
            };
            (
                mask,
                strikes,
                max_pw,
                max_pl,
                f32::NEG_INFINITY,
                probs[(round - round_offset as usize) * num_teams
                    ..(round - round_offset as usize + 1) * num_teams]
                    .iter()
                    .take(max_round as usize)
                    .filter(move |(t, prob)| !prob.is_nan() && (t & mask) == 0)
                    .peekable(),
            )
        };

        // the stack of combinations of teams and strikes to check while searching
        let mut iter_stack = vec![get_iter(curr_mask, mulligans); 1];

        if curr_mask.count_ones() < max_round - 1 {
            iter_stack.extend(
                (0..num_teams)
                    .filter(|&idx| ((1 as TeamsMask) << (idx as u32)) & curr_mask == 0)
                    .flat_map(|idx| {
                        (0..=mulligans.min(1)).map(move |num| {
                            get_iter(
                                curr_mask | ((1 as TeamsMask) << (idx as u32)),
                                mulligans - num,
                            )
                        })
                    }),
            );
        } else {
            (0..num_teams)
                .filter(|&idx| ((1 as TeamsMask) << (idx as u32)) & curr_mask == 0)
                .for_each(|idx| {
                    let mask = ((1 as TeamsMask) << (idx as u32)) | curr_mask;
                    let row = (curr_mask.count_ones() - round_offset) as usize;
                    let p = probs[row * num_teams + idx].1;
                    if mulligans > 0 {
                        max_expected.insert((mask, mulligans), 1.0);
                        max_expected.insert((mask, mulligans - 1), 1.0);
                    } else {
                        max_expected.insert(
                            (mask, mulligans),
                            p + (1.0 - p) * winning_prob[curr_mask.count_ones() as usize],
                        );
                    }
                });
        }

        'outer: while !iter_stack.is_empty() {
            // Safety: garunteed by the while loop check
            let (mask, curr_muls, max_pw, max_pl, max_win_prob, iter) =
                unsafe { iter_stack.last_mut().unwrap_unchecked() };

            // shadow the following so that I can access them without dereferencing
            let mask = *mask;
            let curr_muls = *curr_muls;
            let max_pw = *max_pw;
            let max_pl = *max_pl;

            let round = mask.count_ones();

            if round >= max_round - 1 {
                // then we have reached the end of the number of rounds that are required
                let win_prob = if curr_muls > 0 {
                    1.0
                } else {
                    let prob = iter.next().unwrap().1;
                    prob + (1.0 - prob) * max_pl
                };

                *max_win_prob = win_prob;
            } else if curr_muls > 0 {
                // we have to use `peek` for the inner loop as we don't want to advance the iterator
                // if we have to contiue out of it.
                'inner: while let Some(&&(t, prob)) = iter.peek() {
                    let ideal_win_prob = max_pw * prob + max_pl * (1.0 - prob);
                    // if the win prob in the ideal case reamins below our current best then it's
                    // not going to get better
                    if ideal_win_prob < *max_win_prob {
                        break 'inner;
                    }

                    let win_prob = match (
                        max_expected.get(&(mask | t, curr_muls)),
                        max_expected.get(&(mask | t, curr_muls - 1)),
                    ) {
                        (Some(&win_prob), Some(&lose_prob)) => {
                            win_prob * prob + lose_prob * (1.0 - prob)
                        }
                        (None, Some(&_)) => {
                            iter_stack.push(get_iter(mask | t, curr_muls));
                            continue 'outer;
                        }
                        (Some(&_), None) => {
                            iter_stack.push(get_iter(mask | t, curr_muls - 1));
                            continue 'outer;
                        }
                        (None, None) => {
                            iter_stack.push(get_iter(mask | t, curr_muls));
                            iter_stack.push(get_iter(mask | t, curr_muls - 1));
                            continue 'outer;
                        }
                    };

                    // update the win probability
                    *max_win_prob = max_win_prob.max(win_prob);

                    // progress the iterator
                    iter.next();
                }
            } else {
                // we have to use `peek` for the inner loop as we don't want to advance the iterator
                // if we have to contiue out of it.
                'inner: while let Some(&&(t, prob)) = iter.peek() {
                    let ideal_win_prob = max_pw * prob + max_pl * (1.0 - prob);
                    // if the win prob in the ideal case reamins below our current best then it's
                    // not going to get better
                    if ideal_win_prob < *max_win_prob {
                        break 'inner;
                    }

                    let win_prob = if let Some(&win_prob) = max_expected.get(&(mask | t, curr_muls))
                    {
                        // if `strikes == 0` then `max_pl` is the winning probability if we exit
                        // in the current round
                        win_prob * prob + max_pl * (1.0 - prob)
                    } else {
                        iter_stack.push(get_iter(mask | t, curr_muls));
                        continue 'outer;
                    };

                    // update the win probability
                    *max_win_prob = max_win_prob.max(win_prob);

                    // progress the iterator
                    iter.next();
                }
            }

            max_expected.insert((mask, curr_muls), *max_win_prob);
            iter_stack.pop();
        }
    }

    let mut results: Vec<_> = probs[(curr_mask.count_ones() - round_offset) as usize * num_teams
        ..(curr_mask.count_ones() - round_offset + 1) as usize * num_teams]
        .iter()
        .filter(|&&(idx, _p)| idx & curr_mask == 0)
        .map(|x| {
            (
                x.0.trailing_zeros() as usize,
                (max_expected[&(curr_mask | x.0, mulligans)] * x.1
                    + if mulligans > 0 {
                        max_expected[&(curr_mask | x.0, mulligans - 1)] * (1.0 - x.1)
                    } else {
                        winning_prob[curr_mask.count_ones() as usize] * (1.0 - x.1)
                    }),
                x.1,
            )
        })
        .collect();

    results.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or_else(|| (!a.1.is_nan()).cmp(&!b.1.is_nan()))
            .reverse()
    });

    Ok(results)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let csv_contents =
        fs::read_to_string(args.prob_file_path).context("unable to read the csv file")?;

    let (teams, odds) = read_csv(&csv_contents).context("unable to parse the csv file")?;
    (TeamsMask::BITS as usize >= teams.len())
        .then_some(())
        .with_context(|| {
            format!(
                "too many teams for the size of `TeamsMask`: {}",
                teams.len()
            )
        })?;

    let curr_mask = args
        .teams
        .iter()
        .map(|s1| (s1, teams.iter().position(|s2| s1 == s2)))
        .try_fold(0 as TeamsMask, |acc, (s, idx)| -> Result<_> {
            Ok(acc
                | ((1 as TeamsMask)
                    << idx.with_context(|| format!("unable to find team: {}", s))?))
        })?;

    let num_teams = teams.len();

    // this is your assumed probability for winning if you exit in round n
    let max_round = odds.len() / num_teams + args.starting_round as usize;
    let winning_prob: Vec<_> = (0..=max_round)
        .map(|x| {
            x.saturating_sub(5.min(max_round.saturating_sub(1))) as f32
                / max_round.saturating_sub(5.min(max_round.saturating_sub(1))) as f32
        })
        .map(sstep)
        .collect();

    println!("|{:=^78}|", "");
    println!("|{:^78}|", "Assumed Win Probabilty by Round Exited");
    println!("|{:=^7}|{:=^70}|", "", "");
    winning_prob.chunks(9).enumerate().for_each(|(row, chunk)| {
        print!("| {:<5} |   ", "Round");
        for round in row * 9..row * 9 + chunk.len() {
            print!(" {:^6}", round);
        }
        println!("    {:width$}|", "", width = (9 - chunk.len()) * 7);
        print!("| {:<5} |   ", "Prob.");
        for prob in chunk {
            print!(" {:^6}", format!("{:02.1}%", prob * 100.0));
        }
        println!("    {:width$}|", "", width = (9 - chunk.len()) * 7);
        println!("|{:-^7}|{:-^70}|", "", "");
    });

    let results = solve(
        &odds,
        &winning_prob,
        num_teams,
        curr_mask,
        args.mulligans,
        args.starting_round,
    )
    .context("unable to solve for max win prob")?;

    println!();
    println!("|={:=<62}=|", "");
    println!("| Round {:>55}. |", curr_mask.count_ones() + 1);
    println!("| Mulligans {:>51}. |", args.mulligans);
    println!("|={:=<24}=|={:=^16}=|={:=^16}=|", "", "", "");
    println!(
        "| {:<24} | {:^16} | {:^16} |",
        "Team", "Max win prob (%)", "Game prob (%)"
    );
    println!("|-{:-<24}-|-{:-^16}-|-{:-^16}-|", "", "", "");

    results
        .iter()
        .filter(|x| !x.1.is_nan())
        .for_each(|&(s, pct, g)| {
            println!(
                "| {:<24} | {:^16} | {:^16} |",
                teams[s],
                format!("{:02.5}%", pct * 100.0),
                format!("{:02.5}%", g * 100.0)
            );
        });

    println!("|-{:-<24}-|-{:-^16}-|-{:-^16}-|", "", "", "");

    Ok(())
}
