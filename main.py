import sys
import time
import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import Counter
from math import comb
from vars import *

# import note, strategies only have memory inside a match class
outcomes = []
count_population = 0


def main(
    mass_distribution_name,
    independence_distribution_name,
    mass_weight,
    independence_weight,
    players=PLAYERS,
):
    global outcomes
    global count_population

    if mass_distribution_name not in list(distributions_mass.keys()):
        print(f"The population distribution '{mass_distribution_name}' does not exist.")
        print(
            "Either create the distribution in the vars.py file, or choose from the following:"
        )
        for population in list(distributions_mass.keys()):
            print(population)
        return False

    if independence_distribution_name not in list(distributions_independence.keys()):
        print(
            f"The population distribution '{independence_distribution_name}' does not exist."
        )
        print(
            "Either create the distribution in the vars.py file, or choose from the following:"
        )
        for population in list(distributions_independence.keys()):
            print(population)
        return False

    if len(NUMPY_RANDOM_SEEDS) != len(SEEDS):
        print(
            "The length of population seeds must be equal to the length of tournament seeds."
        )
        return False

    start_time = time.time()

    for i, numpy_seed in enumerate(NUMPY_RANDOM_SEEDS):

        # set player heterogeneity mass and independence and save the players
        set_PLAYER_heterogeneity(
            PLAYERS,
            distributions_mass[mass_distribution_name][count_population],
            distributions_independence[independence_distribution_name][
                count_population
            ],
        )
        save_population_setup(
            mass_distribution_name,
            independence_distribution_name,
            mass_weight,
            independence_weight,
        )

        # save the mass and independence plots
        save_initialized_plot(
            mass_distribution_name,
            independence_distribution_name,
            mass_weight,
            independence_weight,
        )

        # the simulation record
        if count_population == 0:
            print_simulation_record(
                mass_distribution_name,
                independence_distribution_name,
                mass_weight,
                independence_weight,
            )

        # loop over seeds and run simulation
        for _ in range(len(SEEDS)):
            SEED = SEEDS[i]
            print(f"Running seed {SEED}...")
            mp = massBasedMoranProcess(
                PLAYERS,
                match_class=massBasedMatch,
                turns=TURNS,
                seed=SEED,
                mutation_rate=MUTATION_RATE,
                noise=NOISE,
            )

            # loop over moran process until a single strategy dominates the population or max round is reached
            for i, _ in enumerate(mp):
                if (len(mp.population_distribution()) == 1) or (i == MAX_ROUNDS - 1):
                    break

            rounds_played = i
            # save population distribution
            pd.DataFrame(mp.populations).fillna(0).astype(int).to_csv(
                "results/population_evolution/seed_ "
                + str(SEED)
                + "_mass_"
                + str(mass_distribution_name)
                + "_independence_"
                + str(independence_distribution_name)
                + "_mass_weight_"
                + str(mass_weight)
                + "_independence_weight_"
                + str(independence_weight)
                + "_population_seed_"
                + str(numpy_seed)
                + ".csv"
            )

            # save outcomes of each round
            df_outcomes = (
                pd.DataFrame(outcomes)
                .fillna(0)
                .rename(
                    columns={
                        "CC": "coop",
                        "CD": "exploit",
                        "DC": "exploit_",
                        "DD": "defect",
                    }
                )
            )
            df_outcomes["round"] = np.repeat(
                [i + 1 for i in range(rounds_played + 1)], comb(len(PLAYERS), 2)
            )
            df_outcomes["seed"] = SEED
            df_outcomes = df_outcomes.groupby(["round", "seed"]).sum()
            df_outcomes = df_outcomes.astype(int)
            df_outcomes.to_csv(
                "results/outcomes_per_round/seed_"
                + str(SEED)
                + "_mass_"
                + str(mass_distribution_name)
                + "_independence_"
                + str(independence_distribution_name)
                + "_mass_weight_"
                + str(mass_weight)
                + "_independence_weight_"
                + str(independence_weight)
                + "_outcomes_"
                + "population_seed_"
                + str(numpy_seed)
                + ".csv"
            )
            outcomes = []
            break

        count_population += 1

    # show how long simulations took
    print(f"Program ran for {round((time.time() - start_time) / 3600,2)} hours.")


def set_PLAYER_heterogeneity(
    PLAYERS, masses, independences, ids=[i for i in range(len(PLAYERS))]
):
    """
    This functions creates a heterogenous population by adding mass and independence to the player object.
    The object characteristics are used to calculate final scores in the massBaseMatch object.
    """

    for PLAYER, id, mass, independence in zip(PLAYERS, ids, masses, independences):
        setattr(PLAYER, "id", id + 1)
        setattr(PLAYER, "mass", mass)
        setattr(PLAYER, "independence", independence)


def save_initialized_plot(
    mass_distribution_name,
    independence_distribution_name,
    mass_weight,
    independence_weight,
):
    # save the mass and independence plot
    plt.hist(distributions_mass[mass_distribution_name][count_population])
    plt.savefig(
        "results/figures/mass/"
        + str(mass_distribution_name)
        + "_mass_distribution_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".pdf"
    )
    plt.clf()
    plt.hist(
        distributions_independence[independence_distribution_name][count_population]
    )
    plt.savefig(
        "results/figures/independence/"
        + str(independence_distribution_name)
        + "_independence_distribution_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".pdf"
    )
    plt.clf()
    print(
        f"Mass and independence histograms saved from seed: {NUMPY_RANDOM_SEEDS[count_population]}."
    )


def save_population_setup(
    mass_distribution_name,
    independence_distribution_name,
    mass_weight,
    independence_weight,
):
    data = {
        "player_id": [player.id for player in PLAYERS],
        "player_strategy": [player for player in PLAYERS],
        "mass": [player.mass for player in PLAYERS],
        "independence": [player.independence for player in PLAYERS],
        "ratio": [round(player.mass * player.independence, 2) for player in PLAYERS],
    }

    df = pd.DataFrame(data=data)
    df.to_csv(
        "results/population_setup/"
        + "mass_"
        + str(mass_distribution_name)
        + "_independence_"
        + str(independence_distribution_name)
        + "_POPULATION_SETUP_"
        + "population_seed_"
        + str(NUMPY_RANDOM_SEEDS[count_population])
        + ".csv"
    )
    print("Population setup saved.")


def print_simulation_record(
    mass_distribution_name,
    independence_distribution_name,
    mass_weight,
    independence_weight,
):
    print("-" * 75)
    print("\tStarting simulations with the following parameters:")
    print(f"\tMax rounds: {MAX_ROUNDS}")
    print(f"\tTurns: {TURNS}")
    print(f"\tSeeds: {[seed for seed in SEEDS]}")
    print(f"\tPopulations: {len(NUMPY_RANDOM_SEEDS)}")
    print(f"\tMutation rate: {MUTATION_RATE}")
    print(f"\tNoise: {NOISE}")
    print(f"\tmass: {mass_distribution_name} distribution")
    print(f"\tindependence: {independence_distribution_name} distribution")
    print(f"\tmass weight: {mass_weight}")
    print(f"\tindependence weight: {independence_weight}")
    print(f"\tNumber of players: {len(PLAYERS)}")
    print(f"\tStrategies:")
    for strategy in STRATEGIES:
        print(f"\t\t{strategy()}")
    print("-" * 75)


class massBasedMatch(axl.Match):
    """Axelrod Match object with a modified final score function to enable mass to influence the final score as a multiplier"""

    def __init__(
        self, players, turns, seed, noise, mass_weight, independence_weight, **kwargs
    ):
        super().__init__(players=players, turns=turns, seed=seed, noise=noise, **kwargs)
        self.mass_weight, self.independence_weight = mass_weight, independence_weight

    def final_score_per_turn(self):
        outcomes.append(Counter([regex.sub("", str(i)) for i in self.result]))
        base_scores = axl.Match.final_score_per_turn(self)
        mass_scores = [
            PLAYER.mass * score * self.mass_weight
            for PLAYER, score in zip(self.players[::-1], base_scores)
        ]  # list reversed so opponent profits from mass
        return [
            score + (PLAYER.mass * PLAYER.independence * self.independence_weight)
            for PLAYER, score in zip(self.players, mass_scores)
        ]  # list not reversed so player profits from his mass * independence


class massBasedMoranProcess(axl.MoranProcess):
    """Axelrod MoranProcess class"""

    def __init__(
        self,
        players,
        turns,
        seed,
        mutation_rate,
        noise,
        mass_weight,
        independence_weight,
        mass_distribution_name,
        independence_distribution_name,
    ):
        class MatchClass(massBasedMatch):
            def __init__(self, *args, **kwargs):
                super().__init__(
                    *args,
                    mass_weight=mass_weight,
                    independence_weight=independence_weight,
                    **kwargs,
                )

        super().__init__(
            players=players,
            match_class=MatchClass,
            turns=turns,
            seed=seed,
            mutation_rate=mutation_rate,
            noise=noise,
        )
        self.mass_distribution_name = mass_distribution_name
        self.independence_distribution_name = independence_distribution_name

    def __next__(self):
        set_PLAYER_heterogeneity(
            self.players,
            distributions_mass[self.mass_distribution_name][count_population],
            distributions_independence[self.independence_distribution_name][
                count_population
            ],
        )
        super().__next__()
        return self


if __name__ == "__main__":
    # check if the script was called correctly
    if len(sys.argv) < 5:
        print("You have not inputted all necessary terminal arguments.")
    else:
        (
            mass_distribution_name,
            independence_distribution_name,
            mass_weight,
            independence_weight,
        ) = sys.argv[1:]
        if "--explore" in sys.argv:
            repetitions = 50
            for repetition in range(repetitions):
                strategies = random.sample(axl.strategies, 4)
                number_of_players = 10
                players = [player() for player in range(number_of_players) for player in strategies]
                main(mass_distribution_name, players=players)
        else:
            main(mass_distribution_name)
