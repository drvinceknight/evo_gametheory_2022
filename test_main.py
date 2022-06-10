import axelrod as axl

import main


def test_set_PLAYER_Heterogeneity():
    players = axl.Cooperator(), axl.Defector()
    masses = 1.5, 2
    independences = 3, 2
    main.set_PLAYER_heterogeneity(
        PLAYERS=players, masses=masses, independences=independences
    )
    for player, mass, independence, id_ in zip(
        players, masses, independences, range(1, 3)
    ):
        assert player.mass == mass
        assert player.independence == independence
        assert player.id == id_


def test_global_outcomes_at_start():
    assert main.outcomes == []
    assert main.count_population == 0


def test_massBasedMatch():
    assert main.outcomes == []
    players = [axl.TitForTat(), axl.Random(0.3)]
    players[0].mass = 5
    players[1].mass = 3
    players[0].independence = 10
    players[1].independence = 23
    match = main.massBasedMatch(
        players=players,
        turns=20,
        seed=0,
        noise=0.1,
        mass_weight=0.4,
        independence_weight=0.2,
    )
    match.play()
    match.final_score_per_turn()
    assert [{'CD': 7, 'DD': 7, 'DC': 6}] == main.outcomes
    players = [axl.TitForTat(), axl.Alternator()]
    players[0].mass = 5
    players[1].mass = 3
    players[0].independence = 10
    players[1].independence = 23
    match = main.massBasedMatch(
        players=players,
        turns=20,
        seed=0,
        noise=0.1,
        mass_weight=0.4,
        independence_weight=0.2,
    )
    match.play()
    match.final_score_per_turn()
    assert len(main.outcomes) == 2
    assert {'CC': 2, 'CD': 8, 'DD': 1, 'DC': 9} == main.outcomes[1]


def test_massBasedMoranProcess():
    main.outcomes = []
    assert main.outcomes == []
    players = [axl.TitForTat(), axl.Random(0.3), axl.Cooperator()]
    players[0].mass = 5
    players[1].mass = 3
    players[2].mass = 2
    players[0].independence = 10
    players[1].independence = 23
    players[2].independence = 29
    mp = main.massBasedMoranProcess(
        players=players,
        turns=20,
        seed=0,
        noise=0.1,
        mass_weight=0.4,
        independence_weight=0.2,
        mass_distribution_name="normal",
        independence_distribution_name="pareto",
        mutation_rate=.1,
    )
    next(mp)
    assert len(main.outcomes) == 3
    next(mp)
    assert len(main.outcomes) == 6
