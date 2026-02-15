"""
Tests for distributed agentic team architecture.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import pytest

from moe_framework import AgenticAITeam, Objection, TeamReport


def test_default_agentic_team_structure():
    team = AgenticAITeam.create_default()
    team.validate_structure()

    assert team.total_agents == 13
    assert [t.name for t in team.teams] == [
        'Formal Integrity Team',
        'Physical Consistency Team',
        'Implementation & Code Team',
    ]


def test_ceo_verdict_approved():
    team = AgenticAITeam.create_default()
    verdict = team.ceo_verdict([
        TeamReport('Formal Integrity Team', 'Mathematically sound'),
        TeamReport('Physical Consistency Team', 'Physically consistent'),
        TeamReport('Implementation & Code Team', 'Safe merge'),
    ])
    assert verdict['verdict'] == 'APPROVED'
    assert verdict['dominant_objection_team'] is None


def test_ceo_verdict_approved_with_conditions():
    team = AgenticAITeam.create_default()
    verdict = team.ceo_verdict([
        TeamReport('Formal Integrity Team', 'Conditional', [
            Objection('Hidden assumption found', invariant='P_i K = K P_i'),
        ]),
        TeamReport('Physical Consistency Team', 'Physically consistent'),
        TeamReport('Implementation & Code Team', 'Safe merge'),
    ])
    assert verdict['verdict'] == 'APPROVED WITH CONDITIONS'
    assert verdict['dominant_objection_team'] == 'Formal Integrity Team'


def test_ceo_verdict_revise_for_structural_risk():
    team = AgenticAITeam.create_default()
    verdict = team.ceo_verdict([
        TeamReport('Formal Integrity Team', 'Mathematically sound'),
        TeamReport('Physical Consistency Team', 'Physically consistent'),
        TeamReport('Implementation & Code Team', 'Structural risk', [
            Objection('Sector mixing detected', file='moe_framework/moe_system.py'),
        ]),
    ])
    assert verdict['verdict'] == 'REVISE'
    assert verdict['dominant_objection_team'] == 'Implementation & Code Team'


def test_objection_requires_reference():
    team = AgenticAITeam.create_default()
    with pytest.raises(ValueError):
        team.ceo_verdict([
            TeamReport('Formal Integrity Team', 'Conditional', [
                Objection('No concrete reference'),
            ]),
            TeamReport('Physical Consistency Team', 'Physically consistent'),
            TeamReport('Implementation & Code Team', 'Safe merge'),
        ])
