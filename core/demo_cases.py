from __future__ import annotations

from dataclasses import dataclass

from core.news_search import article_id_for
from core.schemas import Article


@dataclass(frozen=True)
class DemoCase:
    slug: str
    title: str
    subject: str
    expected_label: str
    description: str
    articles: tuple[Article, ...]


def _article(title: str, source: str, text: str) -> Article:
    url = f"demo://{source.lower().replace(' ', '-')}/{title.lower().replace(' ', '-')}"
    return Article(
        id=article_id_for(url),
        title=title,
        url=url,
        source=source,
        text=text,
    )


DEMO_CASES: tuple[DemoCase, ...] = (
    DemoCase(
        slug="climate-policy",
        title="Climate policy funding fight",
        subject="climate bill last week",
        expected_label="Mixed",
        description="A policy story with public-investment framing on one side and taxpayer-cost framing on the other.",
        articles=(
            _article(
                "Climate package framed as public investment",
                "Demo Civic Wire",
                (
                    "Supporters framed the climate bill as a public investment in workers, union jobs, clean energy, "
                    "and environmental justice. They argued that the plan protects communities from corporate greed "
                    "while funding grid upgrades and public transit."
                ),
            ),
            _article(
                "Climate package criticized as taxpayer burden",
                "Demo Market Ledger",
                (
                    "Opponents called the climate bill a taxpayer burden and warned that government overreach would "
                    "raise energy prices for small business owners. They said the plan gives agencies too much freedom "
                    "to pick favored industries."
                ),
            ),
            _article(
                "State analysts outline climate bill tradeoffs",
                "Demo Policy Desk",
                (
                    "State analysts said the climate bill could lower long-run infrastructure risk while increasing "
                    "near-term compliance costs. The report listed emission targets, utility spending, and household "
                    "rebate mechanics without endorsing either side."
                ),
            ),
        ),
    ),
    DemoCase(
        slug="border-enforcement",
        title="Border enforcement proposal",
        subject="border bill today",
        expected_label="Lean Right",
        description="A law-and-order framing case with repeated taxpayer, border, and federal-mandate cues.",
        articles=(
            _article(
                "Governor backs border enforcement package",
                "Demo Capitol Beat",
                (
                    "The governor framed the border bill as a law and order response to illegal immigration and "
                    "federal government overreach. Supporters said taxpayers need protection and local sheriffs need "
                    "freedom to cooperate with federal officers."
                ),
            ),
            _article(
                "Small business coalition supports border bill",
                "Demo Commerce Daily",
                (
                    "A small business coalition said the border plan would reduce uncertainty for employers and "
                    "protect communities from unfunded mandates. Critics said the proposal could invite litigation "
                    "and chill immigrant reporting of workplace abuse."
                ),
            ),
        ),
    ),
    DemoCase(
        slug="labor-contract",
        title="Labor contract negotiation",
        subject="transit labor contract",
        expected_label="Lean Left",
        description="A labor story with worker, union, equity, and public-service emphasis.",
        articles=(
            _article(
                "Transit workers push for safer staffing",
                "Demo Labor Journal",
                (
                    "Union leaders said the transit contract should protect workers from unsafe schedules and invest "
                    "in public service reliability. Riders groups backed staffing increases as an equity measure for "
                    "neighborhoods with limited transportation options."
                ),
            ),
            _article(
                "City weighs cost of transit contract",
                "Demo Metro Ledger",
                (
                    "Budget staff said the transit contract would require new public investment but could reduce "
                    "turnover and service disruptions. Business groups asked the city to publish a clearer tax plan "
                    "before approving the agreement."
                ),
            ),
        ),
    ),
    DemoCase(
        slug="sports-schedule",
        title="Non-political sports schedule",
        subject="football schedule update",
        expected_label="Center",
        description="A neutral control case with no policy or ideological framing.",
        articles=(
            _article(
                "Football club announces revised match schedule",
                "Demo Sports Desk",
                (
                    "The football club announced a revised stadium schedule after the tournament calendar changed. "
                    "The match dates, ticket windows, and travel guidance were listed without policy claims or "
                    "ideological language."
                ),
            ),
            _article(
                "Broadcast partners confirm tournament windows",
                "Demo Match Report",
                (
                    "Broadcast partners confirmed kickoff times for the tournament and said streaming details would "
                    "be posted next week. The update focused on logistics, venues, and supporter travel."
                ),
            ),
        ),
    ),
)


def demo_case_titles() -> list[str]:
    return [case.title for case in DEMO_CASES]


def get_demo_case(title: str) -> DemoCase:
    for case in DEMO_CASES:
        if case.title == title or case.slug == title:
            return case
    return DEMO_CASES[0]
