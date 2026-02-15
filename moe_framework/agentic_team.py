"""
Agentic team architecture for distributed, domain-specific review.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


Verdict = Literal['APPROVED', 'APPROVED WITH CONDITIONS', 'REVISE', 'REJECT']


@dataclass(frozen=True)
class Objection:
    """A concrete objection tied to a specific reference."""
    message: str
    equation: Optional[str] = None
    file: Optional[str] = None
    invariant: Optional[str] = None
    failure_case: Optional[str] = None

    def has_reference(self) -> bool:
        return any([self.equation, self.file, self.invariant, self.failure_case])


@dataclass
class TeamReport:
    """Structured supervisor report for a single team."""
    team_name: str
    status: str
    objections: List[Objection] = field(default_factory=list)

    def validate(self) -> None:
        for objection in self.objections:
            if not objection.has_reference():
                raise ValueError(
                    f"Objection in {self.team_name} must include equation, file, "
                    "invariant, or failure_case."
                )


@dataclass(frozen=True)
class TeamDefinition:
    """Definition of a team with three functional agents and one supervisor."""
    name: str
    purpose: str
    agents: List[str]
    supervisor: str


@dataclass
class AgenticAITeam:
    """Three orthogonal teams plus a CEO synthesis agent."""
    teams: List[TeamDefinition]
    ceo_agent: str = 'CEO Agent'
    max_debate_cycles: int = 2

    @classmethod
    def create_default(cls) -> 'AgenticAITeam':
        """Create the required 3x(3 agents + 1 supervisor) + 1 CEO architecture."""
        return cls(
            teams=[
                TeamDefinition(
                    name='Formal Integrity Team',
                    purpose='Mathematical correctness and invariant preservation',
                    agents=['Constructor', 'Skeptic', 'Proof Verifier'],
                    supervisor='Formal Integrity Supervisor',
                ),
                TeamDefinition(
                    name='Physical Consistency Team',
                    purpose='Physical constraint consistency and emergence checks',
                    agents=['Symmetry Analyst', 'Curvature Analyst', 'Effective Field Translator'],
                    supervisor='Physical Consistency Supervisor',
                ),
                TeamDefinition(
                    name='Implementation & Code Team',
                    purpose='Repository alignment with theory and invariant safety',
                    agents=['Researcher', 'Coder', 'Critic'],
                    supervisor='Implementation & Code Supervisor',
                ),
            ]
        )

    def validate_structure(self) -> None:
        """Validate role differentiation and architecture constraints."""
        if len(self.teams) != 3:
            raise ValueError("Architecture must contain exactly 3 teams.")

        for team in self.teams:
            if len(team.agents) != 3:
                raise ValueError(f"{team.name} must contain exactly 3 functional agents.")
            if not team.supervisor:
                raise ValueError(f"{team.name} must define a supervisor.")

        if self.max_debate_cycles > 2:
            raise ValueError("Maximum two debate cycles are allowed per issue.")

    @property
    def total_agents(self) -> int:
        """Total count: 3 teams x (3 agents + 1 supervisor) + 1 CEO = 13."""
        return sum(len(team.agents) + 1 for team in self.teams) + 1

    def ceo_verdict(self, reports: List[TeamReport]) -> Dict[str, Optional[str]]:
        """
        Synthesize supervisor reports without introducing new arguments.
        Returns verdict with dominant objection team when applicable.
        """
        if len(reports) != 3:
            raise ValueError("CEO synthesis requires one report from each of 3 teams.")

        report_map = {report.team_name: report for report in reports}
        for required in [team.name for team in self.teams]:
            if required not in report_map:
                raise ValueError(f"Missing supervisor report for {required}.")

        for report in reports:
            report.validate()

        statuses = {report.team_name: report.status.lower() for report in reports}
        dominant_team = next((r.team_name for r in reports if r.objections), None)

        if any(s in {'invalid', 'violates x'} for s in statuses.values()):
            verdict: Verdict = 'REJECT'
        elif any(s in {'structural risk'} for s in statuses.values()):
            verdict = 'REVISE'
        elif any(s in {'conditional', 'needs constraint'} for s in statuses.values()):
            verdict = 'APPROVED WITH CONDITIONS'
        elif statuses == {
            'Formal Integrity Team': 'mathematically sound',
            'Physical Consistency Team': 'physically consistent',
            'Implementation & Code Team': 'safe merge',
        }:
            verdict = 'APPROVED'
        else:
            verdict = 'REVISE'

        return {
            'verdict': verdict,
            'dominant_objection_team': dominant_team,
        }
