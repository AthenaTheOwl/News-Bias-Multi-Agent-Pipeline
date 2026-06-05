from __future__ import annotations

from core.news_search import article_id_for
from core.schemas import Article


def article(title: str, text: str, url: str = "fixture://article") -> Article:
    return Article(id=article_id_for(url + title), title=title, url=url + "/" + title.replace(" ", "-"), source="fixture", text=text)


LEFT_ARTICLE = article(
    "Climate bill expands public investment",
    (
        "Lawmakers praised the climate package as a public investment in workers, clean energy, and environmental justice. "
        "Supporters said the plan would protect communities from corporate greed and accelerate union jobs. "
        "Opponents questioned the cost but the article emphasized equity and climate action throughout."
    ),
)

RIGHT_ARTICLE = article(
    "Border bill emphasizes law and order",
    (
        "The governor framed the border bill as a law and order response to illegal immigration and government overreach. "
        "Supporters said taxpayers need protection and small business owners need freedom from federal mandates. "
        "Critics argued the plan would invite litigation."
    ),
)

MIXED_ARTICLES = [
    LEFT_ARTICLE,
    RIGHT_ARTICLE,
]

NEUTRAL_ARTICLE = article(
    "Football club announces stadium schedule",
    (
        "The football club announced a revised stadium schedule after the tournament calendar changed. "
        "The match dates and ticket windows were listed without policy claims or ideological language. "
        "Officials said travel guidance would be posted next week."
    ),
)

CHARGED_UNCLEAR_ARTICLE = article(
    "Technology rollout draws sharp language",
    (
        "The rollout was described as reckless by critics and a potential disaster by rival executives. "
        "The article did not connect the dispute to parties, policy ideology, elections, or government programs. "
        "Most claims focused on product timing and customer support."
    ),
)
