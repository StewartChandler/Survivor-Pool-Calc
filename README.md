# Survivor Pool Calc

A simple CLI program for calculating the maximum odds for every pick in a
survivor pool given the odds of each game in the season and the proability of
winning the pool for each round you can exit in.

## Usage

To run the progam, you will need a full list of probabilies of every team
winning their game for every round for the entire season, and then you will need
to put it all in a csv file with the following details:

- Comma separated, unix (LF) or windows (CRLF) line endings, no quote
  characters.
- First row is the team names.
- The `n`th subsequent row is the list of probabilities greater than or equal to
  0, less than or equal to 1 that the teams (repective to the first row) will
  win (or tie) in round `n`.
- A team not playing in given round/week is represented by the an empty entry in
  the row (i.e. nothing between the commas).

For example:

```
team 1, team 2, team 3
0.4, 0.6, 0.2
,0.5,0.3
```

Is the proabilities for a league where there are three teams named *team 1*,
*team 2* and *team 3* respectively. Where, in the first week/round, *team 1* has
a 40% chance of winning, *team 2* has a 60% chance of winning and *team 3* has a
20% chance of winning. In the second week/round we can see that *team 1* is not
playing and that they are not eligable to be picked for the survivor pool.

### Running the program

To run the program you must provide it the path to your csv file.

```posh
cargo run --release -- -p <path-to-your-csv-file>
```

And it will tell you the assumptions made about the probability of you winning
the survivor pool based on an arbitrary function I hardcoded, that kinda looked
right, as well as the maximum probability of you winning for each team you can
pick.

The following is an example output.

```
|==============================================================================|
|                    Assumed Win Probabilty by Round Exited                    |
|=======|======================================================================|
| Round |      0      1      2                                                 |
| Prob. |     0.0%   0.0%  100.0%                                              |
|-------|----------------------------------------------------------------------|

|================================================================|
| Round                                                       1. |
| Mulligans                                                   0. |
|==========================|==================|==================|
| Team                     | Max win prob (%) |  Game prob (%)   |
|--------------------------|------------------|------------------|
| team 1                   |    20.00000%     |    40.00000%     |
| team 2                   |    18.00000%     |    60.00000%     |
| team 3                   |    10.00000%     |    20.00000%     |
|--------------------------|------------------|------------------|
```

I assure you it looks much more impressive with real data.

If we instead run it giving ourselves a single mulligan we can see that we get
different max win probabilities and *team 2* becomes the more optimal pick for
round 1.

```
cargo run --release -- -m=1 -p example.csv
```

As we can see in the following results.

```
|==============================================================================|
|                    Assumed Win Probabilty by Round Exited                    |
|=======|======================================================================|
| Round |      0      1      2                                                 |
| Prob. |     0.0%   0.0%  100.0%                                              |
|-------|----------------------------------------------------------------------|

|================================================================|
| Round                                                       1. |
| Mulligans                                                   1. |
|==========================|==================|==================|
| Team                     | Max win prob (%) |  Game prob (%)   |
|--------------------------|------------------|------------------|
| team 2                   |    72.00000%     |    60.00000%     |
| team 1                   |    70.00001%     |    40.00000%     |
| team 3                   |    60.00000%     |    20.00000%     |
|--------------------------|------------------|------------------|
```

Now, if we simulate picking *team 1* in the first round by passing it as an
argument to the program we can see what it suggests for round 2 (this time
without any mulligans available).

```posh
cargo run --release -- -p example.csv "team 1"
```

Which results in the following output.

```
|==============================================================================|
|                    Assumed Win Probabilty by Round Exited                    |
|=======|======================================================================|
| Round |      0      1      2                                                 |
| Prob. |     0.0%   0.0%  100.0%                                              |
|-------|----------------------------------------------------------------------|

|================================================================|
| Round                                                       2. |
| Mulligans                                                   0. |
|==========================|==================|==================|
| Team                     | Max win prob (%) |  Game prob (%)   |
|--------------------------|------------------|------------------|
| team 2                   |    15.00000%     |    50.00000%     |
|--------------------------|------------------|------------------|
```

You can see that we can only pick *team 2* for round 2 as *team 3* has a bye and
we already picked *team 1*. Additionally, the round displayed is now 2.

Now what if it is already the middle of the season and you only have data for
future games and you want to calculate the optimal choice for the remaining
games you have? Then you can use the `-s=n`, `-s n`, `--starting-round=n`, or
`--starting-round n` flag to indicate that the data you have starts in round
`n`. So, once again we run it, instead this time with `-s=1` indicating that the
first row of proabilities correspods to the second round, and `"team 1"`
indicating that we have selected *team 1* in the first round.

```posh
cargo run --release -- -s=1 -p example.csv "team 1"
```

Which produced the following results.

```
|==============================================================================|
|                    Assumed Win Probabilty by Round Exited                    |
|=======|======================================================================|
| Round |      0      1      2      3                                          |
| Prob. |     0.0%   0.0%   0.0%  100.0%                                       |
|-------|----------------------------------------------------------------------|

|================================================================|
| Round                                                       2. |
| Mulligans                                                   0. |
|==========================|==================|==================|
| Team                     | Max win prob (%) |  Game prob (%)   |
|--------------------------|------------------|------------------|
| team 2                   |    18.00000%     |    60.00000%     |
| team 3                   |    10.00000%     |    20.00000%     |
|--------------------------|------------------|------------------|
```

Where, we can see that although the round is 2, the probabilities used are from
the first row.


## Building

Build and run using cargo:

##### Windows:

```posh
cargo build --release
.\target\release\survivor-pool-calc.exe <args>...
```

##### Everything Else

```sh
cargo build --release
./target/release/survivor-pool-calc <args>...
```

##### Alternatively

```posh
cargo run --release -- <args>...
```
