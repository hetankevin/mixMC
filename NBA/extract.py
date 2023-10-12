import pandas as pd
import numpy as np
import math


# passes = pd.read_csv('passes.csv', nrows=1000)


def extract_trails():
    players = pd.read_csv('NBA/players_ssid.csv')
    players.rename(columns={"ssId": "playerId"}, inplace=True)

    passes = pd.read_csv('NBA/passes.csv')

    props = ["gameId", "season", "period", "shotClock", "ptsScored", "offTeamId", "defTeamId"]
    ix = 0
    grp = passes.groupby("chanceId")
    num_chances = len(grp)
    rows = []

    for _, chance in grp:
        ix += 1

        chance.sort_values("shotClock", inplace=True, ascending=False)
        if (len(chance) <= 1): continue

        chance_start = chance.iloc[[0]]
        chance_start.iloc[0, chance_start.columns.get_loc("receiverId")] = chance_start.iloc[0, chance_start.columns.get_loc("passerId")]
        chance_start.iloc[0, chance_start.columns.get_loc("shotClock")] = 24.0
        chance = pd.concat((chance_start, chance))
        chance.rename(columns={"receiverId": "playerId"}, inplace=True)
        chance["transitionTime"] = chance.shotClock[::-1].diff()[::-1]

        # trail = pd.merge(trail, players, how="left")
        # trail = chance[["startGameClock"]].diff()[::-1].rename(columns={"startGameClock": "transitionTime"})
        # chance_inv = chance[::-1]
        # trail["playerId"] = chance_inv.receiverId
        # trail_start = pd.DataFrame({"transitionTime": 24.0, "playerId": [chance.iloc[-1].passerId]})
        # trail = pd.concat((trail_start, trail))

        trail = pd.merge(chance, players, how="left")
        trail_obj = []
        for row in trail.itertuples():
            trail_obj.append((row.transitionTime, row.playerId, row.name, row.position))

        chance_props = chance.iloc[0][props]
        chance_props["trail"] = trail_obj
        rows.append(chance_props)

        if ix % 100 == 0:
            print(f"{ix} / {num_chances} chances processed: {100 * ix / num_chances:.2f}%")


    df = pd.DataFrame(rows)
    df.to_hdf('NBA/trails.h5', "data")
    return df

# df = extract_trails()


def select_active_trails(df, verbose=False):
    FIRST_N_PLAYERS = 12

    counts = {}
    total_count = 0
    times = {}
    total_time = 0
    for trail in df.trail:
        for (transition_time, player_id, _, _) in trail:
            if math.isnan(transition_time):
                continue
            if not isinstance(player_id, str):
                break
            if player_id not in counts:
                counts[player_id] = 0
                times[player_id] = 0
            counts[player_id] += 1
            total_count += 1
            times[player_id] += transition_time
            total_time += transition_time

    counts = pd.Series(counts).sort_values(ascending=False)
    times = pd.Series(times).sort_values(ascending=False)
    if verbose: print(f"{len(df)} trails: {total_count} passes over {total_time:.2f}s")

    active_players = set(counts[:FIRST_N_PLAYERS].index)

    def filter_trail(trail):
        for (_, player_id, _, _) in trail:
            if player_id not in active_players:
                return False
        return True

    active_df = df.loc[df.trail.apply(filter_trail)]
    
    total_count = 0
    total_time = 0
    for trail in active_df.trail:
        for (transition_time, player_id, _, _) in trail:
            if math.isnan(transition_time):
                continue
            total_count += 1
            total_time += transition_time
    if verbose: print(f"--> selected {len(active_df)} active trails: {total_count} passes over {total_time:.2f}s")

    return active_df


def discretize(df, tau, max_trail_time=20, min_trail_time=10, model_score=False, use_position=False, verbose=False):
    MAX_TRAIL_TIME = max_trail_time
    MIN_TRAIL_TIME = min_trail_time

    def get_player(x):
        _, player_id, _, player_position = x
        player = player_position if use_position else player_id
        if not isinstance(player, str) and math.isnan(player): player = "?"
        return player

    player_ids = {"miss": 0, "score": 1} if model_score else {"idle": 0}
    for trail in df.trail:
        for x in trail:
            player = get_player(x)
            if player not in player_ids:
                player_ids[player] = len(player_ids)

    target_len = int(math.floor(MAX_TRAIL_TIME // tau) + 1)
    
    active_trails_dt = []
    active_trails_ct = []
    ixs = []
    for ix, row in df.iterrows():
        trail = row.trail
        time = 0
        total_time = 0
        trail_dt = []
        trail_ct = []
        for x in trail:
            transition_time, player_id, _, player_position = x
            if math.isnan(transition_time): continue
            # if transition_time < 1e-10: break
            total_time += transition_time
            player = player_ids[get_player(x)]
            trail_ct.append((player, transition_time))
            while time < total_time:
                time += tau
                trail_dt.append(player)
        else:
            if MIN_TRAIL_TIME <= total_time and total_time <= MAX_TRAIL_TIME:
                extend_state = player_ids[("score" if row.ptsScored > 0 else "miss") if model_score else "idle"]
                trail_dt.extend([extend_state] * (target_len - len(trail_dt)))
                active_trails_dt.append(trail_dt)
                active_trails_ct.append(trail_ct)
                # active_trails_ct.append()
                ixs.append(ix)

    if verbose: print(f"--> selected and padded {len(active_trails_dt)} trails")

    # player_ids_inv = {v: k for k, v in player_ids.items()}
    players = pd.read_csv('NBA/players_ssid.csv', index_col="ssId")
    player_ids_inv = {v: (players.loc[k]['name'], players.loc[k]['position']) if k in players.name else k for k, v in player_ids.items()}
    # return np.array(active_trails_dt, dtype=int), , active_trails_ct, player_ids_inv

    active_df = df.loc[ixs]
    active_df["trail_dt"] = active_trails_dt
    active_df["trail_ct"] = active_trails_ct
    return active_df, player_ids_inv


df_all = None

def load_trails(team, season, tau=0.1, max_trail_time=20, min_trail_time=10,
        model_score=False, use_position=False, verbose=False): # team="NYK", season=2022
    global df_all
    if df_all is None: df_all = pd.read_hdf("NBA/trails.h5", "data") 
    selection = df_all[(df_all.season == season) & (df_all.offTeamId == team)]
    if use_position:
        active_trails = selection
    else:
        active_trails = select_active_trails(selection, verbose=verbose)
    return discretize(active_trails, tau=tau, max_trail_time=max_trail_time, min_trail_time=min_trail_time,
                      model_score=model_score, use_position=use_position, verbose=verbose)


# df, player_dict = load_trails(team="NYK", season=2022, model_score=True)
